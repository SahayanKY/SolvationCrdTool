import argparse
import platform
import sys

import numpy as np
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem

def generateConformer(format, value, needAddHs):
    if format == 'SMILES':
        mol = Chem.MolFromSmiles(value)
    elif format == 'MOL':
        mol = Chem.MolFromMolFile(value,removeHs=False)
    elif format == 'MOL2':
        mol = Chem.MolFromMol2File(value,removeHs=False)
    else:
        raise ValueError("invalid choice: {} (choose from 'SMILES', 'MOL', 'MOL2')".format(format))

    if needAddHs:
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
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=numThreads)
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

# 溶液構造を生成
def mksolv(solventconf, soluteconf, solventNum, soluteNum):
    #np.random.seed(seed)
    #
    #分子内の最大距離を算出(溶媒溶質で大きい方を採用)
    #    そのサイズの立方体を並べてそこに回転させた分子を配置していく
    #
    #solventNum+soluteNumの中からsoluteNumの数だけランダムに数を取り出す
    #    その番に溶質を配置する
    #
    #atomPositionList = conf.GetPositions()
    #solventNum+soluteNum回ループ
    #    R = uniform_random_rotateMatrix()
    #    rotated = np.dot(atomPositionList, R)
    #    x,y,z >= 0の境界面にrotatedをくっつけるように平行移動
    #
    #
    #

    return

# ファイルに保存
def saveStructure(structure, saveFilePath):
    return


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

    # 溶媒分子と溶質分子それぞれの単一構造を生成
    solventconf = generateConformer(args.solvent_format, args.solvent, args.solvent_addHs)
    if args.solute is not None:
        soluteconf = generateConformer(args.solute_format, args.solute, args.solute_addHs)
    else:
        soluteconf = None

    # 溶液構造を生成
    structure = mksolv(solventconf, soluteconf, args.solvent_num, args.solute_num)

    # ファイルへ保存
    saveStructure(structure, args.save)



