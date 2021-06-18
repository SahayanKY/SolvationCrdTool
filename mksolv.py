import argparse
import platform
import sys
import os
import itertools
import random
import math

import numpy as np
from scipy.spatial import distance
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem

def generateConformer(format, value, needAddHs):
    if format == 'SMILES':
        mol = Chem.AddHs(Chem.MolFromSmiles(value))
    elif format == 'MOL':
        mol = Chem.MolFromMolFile(value,removeHs=False)
    elif format == 'MOL2':
        mol = Chem.MolFromMol2File(value,removeHs=False)
    else:
        raise ValueError("invalid choice: {} (choose from 'SMILES', 'MOL', 'MOL2')".format(format))

    if format != 'SMILES' and needAddHs:
        # 水素を追加する必要がある場合、
        # 一度SMILESに変換し、molオブジェクトを得てから、addHsする
        mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

    if len(mol.GetConformers()) == 0:
        # conformerが存在しない場合(SMILESを経由している場合)
        # 配座異性体を生成し、最適構造を導く
        # https://www.ag.kagawa-u.ac.jp/charlesy/2020/02/26/2231/

        # numConfsの数の配座異性体を生成
        numConfs = 100
        ps = AllChem.ETKDG()
        ps.pruneRmsThresh = 0 #枝刈りなし
        ps.numThreads = 1

        cids = AllChem.EmbedMultipleConfs(mol, numConfs, ps)
        # それらの構造最適化をする
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
        prop = AllChem.MMFFGetMoleculeProperties(mol)
        # エネルギーを計算し、そのエネルギーのリストを作成する
        energyList = []
        for cid in cids:
            mmff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=cid)
            energyList.append(mmff.CalcEnergy())

        # もっともエネルギーが低かったものを採用する
        conformerID = cids[energyList.index(min(energyList))]
    else:
        conformerID = 0

    return mol.GetConformer(conformerID)

def uniform_random_rotateMatrix():
    # http://bluewidz.blogspot.com/2017/09/blog-post_30.html
    # np.dot(v, R)で変換
    x0 = np.random.random()
    y1 = 2*math.pi*np.random.random()
    y2 = 2*math.pi*np.random.random()
    r1 = math.sqrt(1.0-x0)
    r2 = math.sqrt(x0)
    u0 = math.cos(y2)*r2
    u1 = math.sin(y1)*r1
    u2 = math.cos(y1)*r1
    u3 = math.sin(y2)*r2
    coefi = 2.0*u0*u0-1.0
    coefuu = 2.0
    coefe = 2.0*u0
    R = np.zeros(shape=(3, 3))
    R[0, 0] = coefi+coefuu*u1*u1
    R[1, 1] = coefi+coefuu*u2*u2
    R[2, 2] = coefi+coefuu*u3*u3

    R[1, 2] = coefuu*u2*u3-coefe*u1
    R[2, 0] = coefuu*u3*u1-coefe*u2
    R[0, 1] = coefuu*u1*u2-coefe*u3

    R[2, 1] = coefuu*u3*u2+coefe*u1
    R[0, 2] = coefuu*u1*u3+coefe*u2
    R[1, 0] = coefuu*u2*u1+coefe*u3

    return R

# 分子内の最大距離を算出
def calcMaxInteratomicDistanceIn(conf):
    atomxyzlist = conf.GetPositions()
    return max(distance.pdist(atomxyzlist))

class MolecularGroupBoxIterator():
    def __init__(self, conf, lengthOfGroupBox, time, num):
        time = int(time)
        lengthOfSingleBox = lengthOfGroupBox/time

        self.__conf = conf
        self.__lengthOfGroupBox = lengthOfGroupBox
        self.__time = time
        self.__lengthOfSingleBox = lengthOfSingleBox
        self.__num = num

        self.__atomSymbolList = [atom.GetSymbol() for atom in conf.GetOwningMol().GetAtoms()] * (time*time*time)
        self.__singleBoxDisplacementList = [[i*lengthOfSingleBox, j*lengthOfSingleBox, k*lengthOfSingleBox] for i in range(time) for j in range(time) for k in range(time)]
        self.__singleBoxItr = self.__generateRandomRotatedConformerIter(conf, lengthOfSingleBox, num)


    # conformerを回転させた座標を与えるイテレータを生成
    # 得られる座標値は(paddingも含めて)原点基準にx,y,z>=0に平行移動したもの
    def __generateRandomRotatedConformerIter(self, conf, lengthOfSingleBox, num):
        #conf : rdkit.Chem.Conformer
        #lengthOfSingleBox : 1分子を入れる箱の一辺の長さ
        #num : conformerに指定した化学種の総分子数
        #
        # 全ての分子に対して回転行列を計算するのは無駄なので適当な数に絞る
        rotateMatrixList = [uniform_random_rotateMatrix() for i in range(min(29,num))]

        rotatedList = []
        for R in rotateMatrixList:
            # まず回転させる
            rotated = np.dot(conf.GetPositions(),R)

            # 後の計算を楽にするために、(paddingも含めて)原点基準に平行移動させる
            minCoord = np.min(rotated, axis = 0)
            maxCoord = np.max(rotated, axis = 0)
            width = maxCoord - minCoord
            padding = (lengthOfSingleBox - width)/2
            translated = rotated - minCoord + padding

            rotatedList.append(translated)

        # 各回転行列毎に回転操作を行い、itrに登録
        itr = itertools.cycle(rotatedList)

        return itr # itr.__next__()で順に無限ループで取得

    def __next__(self):
        #GroupBox内の各箱(SingleBox)ごとにrotatedを取得し、
        #displacementを足しこむことでそのSingleBoxに配置する(translated)
        #そしてtranslatedを次々登録していき、最後にそのリストを返す
        groupcoordlist = np.empty((0,3), float)
        for displacement in self.__singleBoxDisplacementList:
            rotated = self.__singleBoxItr.__next__()
            translated = rotated + displacement
            groupcoordlist = np.append(groupcoordlist, translated, axis=0)

        return groupcoordlist

    def __iter__(self):
        return self

    def hasNext(self):
        return True

    def GetAtomSymbols(self):
        return self.__atomSymbolList


# 溶液構造を生成
def mksolv(solventconf, soluteconf, solventNum, soluteNum, saveFilePath):
    solventMaxDist = calcMaxInteratomicDistanceIn(solventconf)
    if soluteconf is None:
        soluteMaxDist = 0
    else:
        soluteMaxDist  = calcMaxInteratomicDistanceIn(soluteconf)

    # 分子間の最低距離
    padding = 1.3 # オングストローム
    # 溶媒1分子、溶質1分子を配置する箱のサイズ
    lengthOfSolventBox = solventMaxDist+padding
    lengthOfSoluteBox  = soluteMaxDist+padding
    # 溶媒に対する溶質の大きさ
    # 溶質が溶媒に対して極端に大きい時、箱1つに溶媒分子1つは無駄なので、複数詰めさせる
    time = max(1.0, math.floor(lengthOfSoluteBox/lengthOfSolventBox))

    # 溶質、溶媒が収まるボックスの数を計算
    # time == 2 の場合、1箱に8分子入る
    # solventNum == 10 の場合、溶媒だけで2箱必要
    groupBoxNum = math.ceil(solventNum / (time*time*time)) + soluteNum
    # 1辺当たりの箱の数を計算
    groupBoxNumPerSide = math.ceil(math.pow(groupBoxNum, 1/3))
    # 箱の辺の長さ
    lengthOfGroupBox = max(lengthOfSolventBox, lengthOfSoluteBox)

    # 溶質を配置するタイミングを決定
    # groupBoxNumの中からsoluteNumの数だけランダムに数を取り出す
    indexSoluteList = random.sample(range(groupBoxNum), k=soluteNum)

    # イテレータ取得
    solventGroupBoxIter = MolecularGroupBoxIterator(solventconf, lengthOfGroupBox, time, solventNum)
    if soluteconf is not None:
        soluteGroupBoxIter = MolecularGroupBoxIterator(soluteconf, lengthOfGroupBox, 1, soluteNum)
    else:
        soluteGroupBoxIter = None

    print('time:{}'.format(time))
    print('groupBoxNum:{}'.format(groupBoxNum))
    print('groupBoxNumPerSide:{}'.format(groupBoxNumPerSide))
    print('indexSoluteList:{}'.format(indexSoluteList))

    i = 0
    j = 0
    k = 0
    minCoord = np.array([ -groupBoxNumPerSide/2 * lengthOfGroupBox ] *3)
    for boxi in range(groupBoxNum):
        boxiMinCoord = minCoord + [i*lengthOfGroupBox, j*lengthOfGroupBox, k*lengthOfGroupBox]

        if boxi in indexSoluteList:
            # 今回は溶質を配置
            groupBoxCoords = soluteGroupBoxIter.__next__()
            groupBoxAtoms  = soluteGroupBoxIter.GetAtomSymbols()
        else:
            # 今回は溶媒を配置
            groupBoxCoords = solventGroupBoxIter.__next__()
            groupBoxAtoms  = solventGroupBoxIter.GetAtomSymbols()

        # groupboxをboxiに配置する
        groupBoxCoords = boxiMinCoord + groupBoxCoords

        # 書き出し
        saveStructure(groupBoxCoords, groupBoxAtoms, saveFilePath)

        # インクリメント
        i = i+1
        if i == groupBoxNumPerSide:
            i = 0
            j = j+1
        if j == groupBoxNumPerSide:
            j = 0
            k = k+1


    return

# ファイルに保存
def saveStructure(coords, atomsymbols, saveFilePath):
    # mode='a' : 末尾に追記
    with open(saveFilePath, mode='a') as f:
        # 最終行の次から書き始める
        f.write('\n')
        contents = '\n'.join([ '{} {} {} {}'.format(symbol,x,y,z) for symbol,(x,y,z) in zip(atomsymbols,coords)])
        f.write(contents)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mksolv : Automatically generate molecular coordinates in solution.')

    parser.add_argument('--solvent_format', help='solvent format', required=True, choices=['SMILES', 'MOL', 'MOL2'])
    parser.add_argument('--solvent', help='solvent value', required=True)
    parser.add_argument('--solvent_num', help='number of solvent', required=True, type=int)
    parser.add_argument('--solvent_addHs', help='add H atoms to solvent', action='store_true')

    parser.add_argument('--solute_format', help='solute format', choices=['SMILES', 'MOL', 'MOL2'])
    parser.add_argument('--solute', help='solute value')
    parser.add_argument('--solute_num', help='number of solute', type=int, default=0)
    parser.add_argument('--solute_addHs', help='add H atoms to solute', action='store_true')

    parser.add_argument('--save', help='filename')

    # 引数解析
    args = parser.parse_args()

    if args.solute_format is None and args.solute is not None:
        parser.error("The '--solute' argument requires the '--solute-format'")
    if args.solute is not None and args.solute_num == 0:
        parser.error("The '--solute_num' argument must be greater than 0")

    # バージョン確認
    print('python:{}'.format(platform.python_version()))
    if sys.version_info[0] != 3:
        print('Warnings:mksolv assumes python3.')
    print('rdkit:{}'.format(rdBase.rdkitVersion))

    # 保存先確認
    if os.path.exists(args.save):
        # 保存先が既に存在していた場合
        raise IOError('{} already exists.'.format(args.save))

    # 溶媒分子と溶質分子それぞれの単一構造を生成
    solventconf = generateConformer(args.solvent_format, args.solvent, args.solvent_addHs)
    if args.solute is not None:
        soluteconf = generateConformer(args.solute_format, args.solute, args.solute_addHs)
    else:
        soluteconf = None

    # TODO seedの取り扱いについて後で考える
    np.random.seed(0)
    random.seed(0)

    # 溶液構造を生成
    structure = mksolv(solventconf, soluteconf, args.solvent_num, args.solute_num, args.save)


