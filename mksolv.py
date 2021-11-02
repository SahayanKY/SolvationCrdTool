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
from rdkit.Chem import Descriptors
from rdkit.Chem import rdchem

class Conformer():
    # TODO 分子量の計算メソッドを追加
    residue_id = 1

    def __init__(self, format, value, needAddHs):
        self.residue_id = Conformer.residue_id
        Conformer.residue_id += 1

        # まずrdkitで読み込めないか試す
        mol = self.__generateRdkitMol(format, value, needAddHs)
        if mol is not None:
            # 正常に読み込めた場合
            # rdkitのConformerを生成する
            rdconf = self.__generateRdkitConformer(mol)

            # 原子名や残基名
            atomlist = rdconf.GetOwningMol().GetAtoms()
            if atomlist[0].GetPDBResidueInfo() is None:
                # PDB情報が無い場合(SMILES等)
                self.__atomNameList = [atom.GetSymbol() for atom in atomlist]
                residueName = 'R{:0>2}'.format(self.residue_id)
                self.__residueNameList = [residueName] * len(atomlist)

            else:
                #原子名
                self.__atomNameList = [atom.GetPDBResidueInfo().GetName().replace(' ', '') for atom in atomlist]
                #残基名
                self.__residueNameList = [atom.GetPDBResidueInfo().GetResidueName().replace(' ', '') for atom in atomlist]

            # 座標値
            self.__positionList = rdconf.GetPositions()

        elif format == 'PDB':
            # EPなどの存在によりrdkitでは読み込めなかった可能性がある場合
            # 改めてPDBを読み込み直す
            self.__atomNameList, self.__residueNameList, self.__positionList = self.__generatePDBConformer(value)


        # 分子量
        periodictable = rdchem.GetPeriodicTable()
        def getAW(symbol):
            s = symbol.strip('0123456789')
            try:
                return periodictable.GetAtomicWeight(s)
            except RuntimeError:
                print('AtomicWeight of {} is 0'.format(symbol))
                return 0
        self.__molwt = sum([getAW(symbol) for symbol in self.__atomNameList])



    def __generateRdkitMol(self, format, value, needAddHs):
        if format == 'SMILES':
            mol = Chem.AddHs(Chem.MolFromSmiles(value))
        elif format == 'MOL':
            mol = Chem.MolFromMolFile(value,removeHs=False)
        elif format == 'MOL2':
            mol = Chem.MolFromMol2File(value,removeHs=False)
        elif format == 'PDB':
            mol = Chem.MolFromPDBFile(value,removeHs=False)
        else:
            raise ValueError("invalid choice: {} (choose from 'SMILES', 'MOL', 'MOL2', 'PDB')".format(format))

        if format != 'SMILES' and needAddHs:
                # 水素を追加する必要がある場合、
                # 一度SMILESに変換し、molオブジェクトを得てから、addHsする
                mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
        return mol


    def __generateRdkitConformer(self, mol):
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

    def __generatePDBConformer(self, pdbFilePath):
        with open(pdbFilePath, mode='r') as f:
            # 原子の部分のみ取り出す
            lines = [s.strip('\n') for s in f.readlines() if s.startswith('ATOM') or s.startswith('HETATM')]

        #1始まりで
        #13-16: 原子名
        #18-20: 残基名
        #31-38: X座標
        #39-46: Y座標
        #47-54: Z座標
        # 原子名
        atomNameList = [s[12:16].replace(' ', '') for s in lines]
        # 残基名
        residueNameList = [s[17:20].replace(' ', '') for s in lines]
        # 座標値
        positionList = np.array([[float(s[30:38]),float(s[38:46]),float(s[46:54])] for s in lines])

        return atomNameList, residueNameList, positionList

    def giveAtomPositions(self):
        # TODO 出来ればディープコピーにしたいが
        return self.__positionList

    def giveAtomNames(self):
        return self.__atomNameList

    def giveResidueNames(self):
        return self.__residueNameList

    def giveMolWt(self):
        return self.__molwt


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
    atomxyzlist = conf.giveAtomPositions()
    return max(distance.pdist(atomxyzlist))

class MolecularGroupBoxIterator():
    iter_id = 1

    def __init__(self, conf, lengthOfGroupBox, time, arrangeNumList):
        time = int(time)
        lengthOfSingleBox = lengthOfGroupBox/time

        self.__conf = conf
        self.__lengthOfGroupBox = lengthOfGroupBox
        self.__time = time
        self.__lengthOfSingleBox = lengthOfSingleBox
        self.__arrangeNumList = arrangeNumList
        self.__arrangeNumListIter = arrangeNumList.__iter__()

        self.__singleBoxDisplacementList = [[i*lengthOfSingleBox, j*lengthOfSingleBox, k*lengthOfSingleBox] for i in range(time) for j in range(time) for k in range(time)]
        self.__singleBoxItr = self.__generateRandomRotatedConformerIter(conf, lengthOfSingleBox)


    # conformerを回転させた座標を与えるイテレータを生成
    # 得られる座標値は(paddingも含めて)原点基準にx,y,z>=0に平行移動したもの
    def __generateRandomRotatedConformerIter(self, conf, lengthOfSingleBox):
        #conf : rdkit.Chem.Conformer
        #lengthOfSingleBox : 1分子を入れる箱の一辺の長さ
        #num : conformerに指定した化学種の総分子数
        #
        # 配置する総分子数
        num = sum(self.__arrangeNumList)
        # 全ての分子に対して回転行列を計算するのは無駄なので適当な数に絞る
        rotateMatrixList = [uniform_random_rotateMatrix() for i in range(min(29,num))]

        rotatedList = []
        for R in rotateMatrixList:
            # まず回転させる
            rotated = np.dot(conf.giveAtomPositions(),R)

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
        # 今回の配置数を取得
        arrangeNum = self.__arrangeNumListIter.__next__()
        # スキップするタイミングを決定
        arrangeIndexList = random.sample(range(self.getMoleculeNum()), k=arrangeNum)

        # GroupBox内の各箱(SingleBox)ごとにrotatedを取得し、
        # displacementを足しこむことでそのSingleBoxに配置する(translated)
        # そしてtranslatedを次々登録していき、最後にそのリストを返す
        groupcoordlist = np.empty((0,3), float)
        for i, displacement in enumerate(self.__singleBoxDisplacementList):
            if i not in arrangeIndexList:
                # 過剰分処理のためここはスキップ
                continue
            rotated = self.__singleBoxItr.__next__()
            translated = rotated + displacement
            groupcoordlist = np.append(groupcoordlist, translated, axis=0)

        # 実際に配置する分子数だけatomnamesやresiduenamesなどを倍増させる
        groupatomnames = self.__singleBoxAtomNameList * arrangeNum
        groupresiduenames = self.__singleBoxResidueNameList * arrangeNum
        groupatomnums = range(len(groupatomnames))
        groupresiduenums = itertools.chain.from_iterable([ [i] * len(self.__singleBoxAtomNameList) for i in range(arrangeNum)])

        return groupcoordlist, groupatomnames, groupatomnums, groupresiduenames, groupresiduenums

    def __iter__(self):
        return self

    def hasNext(self):
        return True

    def getMoleculeNum(self):
        return self.__time * self.__time * self.__time


# 溶液構造を生成
def mksolv(solventconf, soluteconf, solventNum, soluteNum, saveFilePath):
    solventMaxDist = calcMaxInteratomicDistanceIn(solventconf)
    if soluteconf is None:
        soluteMaxDist = 0
    else:
        soluteMaxDist  = calcMaxInteratomicDistanceIn(soluteconf)

    # 分子間の最低距離
    padding = 0.7 # オングストローム
    # 溶媒1分子、溶質1分子を配置する箱のサイズ
    lengthOfSolventBox = solventMaxDist+padding
    lengthOfSoluteBox  = soluteMaxDist+padding
    # 溶媒に対する溶質の大きさ
    # 溶質が溶媒に対して極端に大きい時、箱1つに溶媒分子1つは無駄なので、複数詰めさせる
    time = max(1.0, math.floor(lengthOfSoluteBox/lengthOfSolventBox))

    # 溶質、溶媒が収まるボックスの数を計算
    # time == 2 の場合、1箱に8分子入る
    # solventNum == 10 の場合、溶媒だけで2箱必要(過剰分は配置するときに減らして処理)
    # さらにgroupBoxを立方体に配置できるように三乗根のceilの三乗をとる
    # 1辺当たりの箱の数を計算
    groupBoxNumPerSide = math.ceil(math.pow(solventNum / (time*time*time) + soluteNum, 1/3))
    groupBoxNum = int(math.pow(groupBoxNumPerSide, 3))
    # 過剰の溶媒分子の数を計算し、その分を後で引く
    solventTotalExcessNum = (groupBoxNum-soluteNum) * time*time*time - solventNum
    # 箱の辺の長さ
    lengthOfGroupBox = max(lengthOfSolventBox, lengthOfSoluteBox)

    # 溶質を配置するタイミングを決定
    # groupBoxNumの中からsoluteNumの数だけランダムに数を取り出す
    indexSoluteList = random.sample(range(groupBoxNum), k=soluteNum)

    # 各groupboxにおいて、それぞれで幾つ分子を取り除くかを決定(過剰分の処理)
    # (ただし、溶質は全て過剰ゼロ)
    # まずは溶媒に関して過剰分の平均をとり、最低の過剰数を計算(確実にこの分は差し引く)
    solventAverageExcessNum = math.floor(solventTotalExcessNum / (groupBoxNum-soluteNum))
    solventExcessNumList = [ solventAverageExcessNum ] * (groupBoxNum-soluteNum)
    # 残りは乱数で配分
    for i in random.sample(range(len(solventExcessNumList)), k=int(solventTotalExcessNum - solventAverageExcessNum*(groupBoxNum-soluteNum))):
        solventExcessNumList[i] += 1
    # 各groupboxにおいて実際に配置する数を決定
    solventArrangeNumList = [int(time*time*time - excess) for excess in solventExcessNumList]
    # 溶質を配置するgroupboxでは過剰はゼロ
    soluteArrangeNumList = [1] * soluteNum


    # イテレータ取得
    solventGroupBoxIter = MolecularGroupBoxIterator(solventconf, lengthOfGroupBox, time, solventArrangeNumList)
    if soluteconf is not None:
        soluteGroupBoxIter = MolecularGroupBoxIterator(soluteconf, lengthOfGroupBox, 1, soluteArrangeNumList)
    else:
        soluteGroupBoxIter = None

    print('time:{}'.format(time))
    print('groupBoxNum:{}'.format(groupBoxNum))
    print('groupBoxNumPerSide:{}'.format(groupBoxNumPerSide))
    print('indexSoluteList:{}'.format(indexSoluteList))
    print('solventTotalExcessNum:{}'.format(solventTotalExcessNum))
    # 単位 : g/mol
    soluteMass = 0 if soluteconf is None else soluteconf.giveMolWt() * soluteNum
    solventMass = solventconf.giveMolWt() * solventNum
    # 単位 : angstrom^3
    soluteVolume = soluteNum * math.pow(lengthOfGroupBox,3)
    solventVolume = (groupBoxNum-soluteNum) * math.pow(lengthOfGroupBox,3)
    if soluteconf is not None:
        print('pureSoluteDensity:{:7.5f}[g/cm3]'.format(soluteMass/soluteVolume * (10/6.02214076)))
        print('pureSolventDensity:{:7.5f}[g/cm3]'.format(solventMass/solventVolume * (10/6.02214076)))
    print('solutionDensity:{:7.5f}[g/cm3]'.format((soluteMass+solventMass)/(soluteVolume+solventVolume) * (10/6.02214076)))

    i = 0
    j = 0
    k = 0
    atomNumOffset = 1
    residueNumOffset = 1
    minCoord = np.array([ -groupBoxNumPerSide/2 * lengthOfGroupBox ] *3)
    for boxi in range(groupBoxNum):
        boxiMinCoord = minCoord + [i*lengthOfGroupBox, j*lengthOfGroupBox, k*lengthOfGroupBox]

        if boxi in indexSoluteList:
            # 今回は溶質を配置
            groupboxiter = soluteGroupBoxIter
        else:
            # 今回は溶媒を配置
            groupboxiter = solventGroupBoxIter

        coords, atomNames, atomNums, residueNames, residueNums = groupboxiter.__next__()
        atomNums = [atomNumOffset + i for i in atomNums]
        residueNums = [ residueNumOffset + i for i in residueNums ]

        # groupboxをboxiの場所に配置する
        coords = boxiMinCoord + coords

        # 中身がある場合のみ
        if len(coords) != 0:
            # 書き出し
            saveStructure(coords, atomNames, atomNums, residueNames, residueNums, saveFilePath)
            # オフセット移動
            atomNumOffset = atomNums[-1] +1
            residueNumOffset = residueNums[-1] +1

        # インクリメント
        i = i+1
        if i == groupBoxNumPerSide:
            i = 0
            j = j+1
        if j == groupBoxNumPerSide:
            j = 0
            k = k+1



# ファイルに保存
def saveStructure(coords, atomNames, atomNums, residueNames, residueNums, saveFilePath):
    # mode='a' : 末尾に追記
    with open(saveFilePath, mode='a') as f:
        # 最終行の次から書き始める
        contents = '\n'.join([ 'ATOM  {:>5} {:<4} {:<3}  {:>4}    {:8.3f}{:8.3f}{:8.3f}'.format(anum,aname,rename,renum,x,y,z) for anum, aname, rename, renum,(x,y,z) in zip(atomNums,atomNames,residueNames,residueNums,coords)])
        f.write(contents)
        f.write('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mksolv : Automatically generate molecular coordinates in solution.')

    parser.add_argument('--solvent_format', help='solvent format', required=True, choices=['SMILES', 'MOL', 'MOL2', 'PDB'])
    parser.add_argument('--solvent', help='solvent value', required=True)
    parser.add_argument('--solvent_num', help='number of solvent', required=True, type=int)
    parser.add_argument('--solvent_addHs', help='add H atoms to solvent', action='store_true')

    parser.add_argument('--solute_format', help='solute format', choices=['SMILES', 'MOL', 'MOL2', 'PDB'])
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


