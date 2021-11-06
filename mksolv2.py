import argparse
import platform
import sys
import os
import itertools
import random
import math

import numpy as np
import pandas as pd
from scipy.spatial import distance
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


from mksolv import *

def calcGroupBoxNumPerSide(timeList, numList):
    """
    各Conformerに対して指定されたtime設定と配置数を適用した場合に
    1辺当たりのGroupBoxの数を算出
    """
    df = pd.DataFrame({'time':timeList, 'num':numList})
    # 各time毎にnumの和をとる
    numSum_groupByTime = df.groupby('time')['num'].sum()
    # 実際に必要なのはこの情報なので置き換える
    timeList = numSum_groupByTime.index
    numList = numSum_groupByTime.values

    # GroupBoxを立方体に並べたときの1辺当たりのGroupBoxの数を算出
    # time == 2 の場合、1箱に8分子入る
    # solventNum == 10 の場合、溶媒だけで2箱必要(過剰分は配置するときに減らして処理)
    # さらにgroupBoxを立方体に配置できるように三乗根のceilの三乗をとる
    # 1辺当たりの箱の数を計算
    groupBoxNumPerSide = math.ceil(math.pow(
            sum([math.ceil(n/(t*t*t)) for t,n in zip(timeList,numList)]),
            1/3
        ))

    return groupBoxNumPerSide

# 溶液構造を生成
def mksolv2(confList, numList, saveFilePath):
    # 各Conformerにおいて、分子内距離の最大値を算出
    confMaxDistList = [ calcMaxInteratomicDistanceIn(conf) for conf in confList ]

    # 分子間の最低距離
    padding = 0.7 # オングストローム
    # 各Conformer1分子を配置する箱のサイズ
    lengthOfConformerBoxList = [dist+padding for dist in confMaxDistList]

    # 最も大きい1分子箱のサイズを調べる
    # そのサイズが単一のGroupBoxのサイズになる
    lengthOfGroupBox = max(lengthOfConformerBoxList)
    # 例えば溶質が溶媒に対して極端に大きい時、箱1つに溶媒分子1つは無駄なので、複数詰めさせる
    # その倍率timeの上限を求める
    upperLimitOfTimeList = [ math.floor(lengthOfGroupBox/length) for length in lengthOfConformerBoxList ]

    # 最適なtimeの組合せについて検討
    groupBoxNumPerSide = None
    for candiTimeList in itertools.product(*[range(1,n+1) for n in upperLimitOfTimeList]):
        boxnum = calcGroupBoxNumPerSide(candiTimeList, numList)
        if groupBoxNumPerSide is None or optimizedGroupBoxNumPerSide > boxnum:
            groupBoxNumPerSide = boxnum
            timeList = candiTimeList

    # confとtimeとnumを一つのDictにまとめてしまう
    # {
    #     1: {'conf':[conf1,conf2], 'num':[1,14]},
    #     2: {'conf':[conf3], 'num':[150]}
    # }
    # groupbyを使うために一度DataFrameを経由する
    df = pd.DataFrame({'conf':confList, 'time':timeList, 'num':numList})
    groupdict = {d[0]:d[1][['conf','num']].to_dict(orient='list') for d in df.groupby('time')}

    # 各groupBoxに配置する分子の数を決定する
    # その前に空隙を配置する数を求める
    # 空隙には
    # ・各time毎に複数の立方体に整形することにより発生する空隙
    # ・全体を1つの立方体に整形することにより発生する空隙
    # の2種類が存在する
    # 後者の空隙は最も分子数の多いtimeに埋めてもらうことで対処する
    #
    # 前者の空隙数を設定
    for t in groupdict.keys():
        sumNum = sum(groupdict[t]['num'])
        groupBoxNum_fort = math.ceil(sumNum/(t*t*t))
        voidNum = int(groupBoxNum_fort*(t*t*t) -sumNum)
        groupdict[t]['groupBoxNum'] = groupBoxNum_fort
        groupdict[t]['voidNum'] = voidNum
    # 後者の空隙数を追加
    # その前に現状のgroupboxの数と立方体にするために必要なgroupboxの数を算出
    # その差から必要な空隙の数を算出
    # 現状のgroupboxの数
    currentGroupBoxNum = sum([d[t]['groupBoxNum'] for t in groupdict.keys()])
    # 立方体にするのに必要なgroupboxの数
    groupBoxNum = int(math.pow(groupBoxNumPerSide, 3))
    # どのtimeに空隙を追加するのかを決める
    d1 = {t:sum(groupdict[t]['num']) for t in groupdict.keys()}
    timeOfMaxNum = max(d1, key=d1.get)
    # 後者の空隙を追加
    groupdict[timeOfMaxNum]['voidNum'] += (groupBoxNum-currentGroupBoxNum)*(timeOfMaxNum*timeOfMaxNum*timeOfMaxNum)

    # TODO ここから修正再開
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
    # 溶質を配置するgroupboxでは過剰はゼロ
    soluteExcessNumList = [0] * soluteNum

    # イテレータ取得
    solventGroupBoxIter = MolecularGroupBoxIterator(solventconf, lengthOfGroupBox, time, solventExcessNumList)
    if soluteconf is not None:
        soluteGroupBoxIter = MolecularGroupBoxIterator(soluteconf, lengthOfGroupBox, 1, soluteExcessNumList)
    else:
        soluteGroupBoxIter = None

    print('time:{}'.format(time))
    print('groupBoxNum:{}'.format(groupBoxNum))
    print('groupBoxNumPerSide:{}'.format(groupBoxNumPerSide))
    print('indexSoluteList:{}'.format(indexSoluteList))
    print('solventTotalExcessNum:{}'.format(solventTotalExcessNum))
    # 単位 : g/mol
    soluteMass = 0 if soluteconf is None else Descriptors.MolWt(soluteconf.GetOwningMol()) * soluteNum
    solventMass = Descriptors.MolWt(solventconf.GetOwningMol()) * solventNum
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


if __name__ == '__main__':
    # バージョン確認
    print('python:{}'.format(platform.python_version()))
    if sys.version_info[0] != 3:
        print('Warnings:mksolv assumes python3.')
    print('rdkit:{}'.format(rdBase.rdkitVersion))


    argv = sys.argv
    # 引数解釈には使わないがチェックやhelpのために使う
    parser = argparse.ArgumentParser(description='mksolv2 : Automatically generate molecular coordinates in solution.')

    parser.add_argument('--save', help='filename', required=True)
    parser.add_argument('--mol', help='<format> <molecule> <num> [needAddHs] (if need, specify this option many times.)', required=True, nargs='+')

    # チェック
    parser.parse_args()

    # 引数解釈
    # argvの第1要素はこのスクリプトファイル名なので除去
    argv.pop(0)
    # 引数に--があるインデックスを探し、
    # それぞれで分割して解釈
    optionIndexes = [i for i,s in enumerate(argv) if '--' in s] + [len(argv)]
    confList = []
    numList = []
    for start, end in zip(optionIndexes[0:],optionIndexes[1:]):
        if argv[start] == '--save':
            if end - start != 2:
                # 値の数を確認
                parser.error("--save argument requires FilePath")
            saveFilePath = argv[start+1]
            # 保存先確認
            if os.path.exists(saveFilePath):
                # 保存先が既に存在していた場合
                raise IOError('{} already exists.'.format(saveFilePath))
        elif argv[start] == '--mol':
            if end - start != 4 or end - start != 5:
                # 値の数を確認
                parser.error("--mol argument requires <format> <molecule> <num> [needAddHs]")
            format = argv[start+1]
            molecule = argv[start+2]
            num = argv[start+3]
            if not num.isdecimal() or int(num) < 1:
                parser.error("<num> must be int and greater than 0")
            else:
                num = int(num)
            if end - start == 5:
                if type(argv[start+4]) is bool:
                    needAddHs = argv[start+4]
                else:
                    parser.error("[needAddHs] must be boolean")
            else:
                needAddHs = False

            # 分子の単一構造を生成し保存
            conf = generateConformer(format, molecule, needAddHs)
            confList.append(conf)
            numList.append(num)


    # TODO seedの取り扱いについて後で考える
    np.random.seed(0)
    random.seed(0)

    # 溶液構造を生成
    structure = mksolv2(confList, numList, saveFilePath)








