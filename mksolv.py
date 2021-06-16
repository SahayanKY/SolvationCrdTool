import argparse
import platform
import sys

from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem

def generateConformer(format,value):

    #mol = Chem.MolFromMol2File(mol2FilePath)
    #mol = Chem.MolFromSmiles(smiles)
    return

def mksolv(solventconf, soluteconf, solventNum, soluteNum):
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mksolv : Automatically generate molecular coordinates in solution.')

    # TODO addHsフラグオプションを入れる
    # trueならばSMILESに変換してコンフォマーを得る
    # falseならば得られたmolオブジェクトから直接コンフォマーを得る
    parser.add_argument('--solvent_format', help='solvent format', required=True, choices=['SMILES', 'MOL', 'MOL2'])
    parser.add_argument('--solvent', help='solvent value', required=True)
    parser.add_argument('--solvent_num', help='number of solvent', required=True, type=int)
    parser.add_argument('--solvent_addHs', help='add H atoms to solvent', action='store_true')

    parser.add_argument('--solute_format', help='solute format', choices=['SMILES', 'MOL', 'MOL2'])
    parser.add_argument('--solute', help='solute value')
    parser.add_argument('--solute_num', help='number of solute', type=int, default=1)
    parser.add_argument('--solute_addHs', help='add H atoms to solute', action='store_true')

    parser.add_argument('--save', help='filename')

    # 引数解析
    args = parser.parse_args()

    # TODO 要検証(Noneになるはずなのでこれで判定できるか不明)
    if ('solute' in vars(args) and
        'solute-format' not in vars(args)):
        parser.error("The '--solute' argument requires the '--solute-format'")

    # バージョン確認
    print('python:{}'.format(platform.python_version()))
    if sys.version_info[0] != 3:
        print('Warnings:mksolv assumes python3.')
    print('rdkit:{}'.format(rdBase.rdkitVersion))

    # 溶媒分子と溶質分子それぞれの単一構造を生成
    solventconf = generateConformer(args.solvent_format, args.solvent)
    soluteconf = generateConformer(args.solute_format, args.solute)

    # 溶液構造を生成
    mksolv(solventconf, soluteconf, args.solvent_num, args.solute_num)