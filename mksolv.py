import argparse
import platform
import sys

from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem

def generateConformer(format, value, needAddHs):
    if format == 'SMILES':
        mol = Chem.MolFromSmiles(value)
    elif format == 'MOL':
        mol = Chem.MolFromMolFile(value)
    elif format == 'MOL2':
        mol = Chem.MolFromMol2File(value)
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


# 溶液構造を生成
def mksolv(solventconf, soluteconf, solventNum, soluteNum):
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



