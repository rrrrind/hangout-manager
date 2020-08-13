import os

import numpy as np
import pandas as pd

class DataGenerator(object):
    """
    回答結果から各種学習データセットを作成するクラス
    
    Example
    -------
    # データセットの生成
    >>> from data_generator import DataGenerator
    >>> dg = DataGenrator('<回答結果のCSVファイルへのパス')
    >>> dg.generate()
    # データセットへのアクセス
    >>> dg.df_static_info_binary   # 静的情報（バイナリベクトル）
    >>> dg.df_static_info_weight   # 静的情報（重み付きベクトル）
    >>> dg.df_dynamic_info         # 動的情報
    >>> dg.df_human_info           # 性格情報
    """
    
    def __init__(self, answer_path):
        
        # 遊びのジャンル一覧（未回答は"なし"に変換）
        self.GENRE_LIST = ['アウトドア（マリンスポーツ・釣り・BBQ・登山・ハイキングなど）', 
                           'エンタメ（ボウリング場・カラオケ・ゲームセンターなど）', 
                           'スポーツ（ボルダリング・ゴルフ・サイクリング・ジム・ヨガなど）',
                           '観光（観光名所巡り・日帰り温泉・食べ歩きなど）',
#                            '遊園地・動物園・水族館・海水浴場',
                           '友達との飲み会（外飲み・飲み歩きなど）',
                           '外での買い物（アウトレットなど）',
                           'ショー・展示・催し（映画館・ミュージカル・博物館・美術館など）',
                           'なし']
        
        # 回答結果
        self.df_answer = pd.read_csv(answer_path)
        # 回答結果を前処理
        def preprocessing():
            # 明後日の予定の質問の選択肢の連番が4と5で逆になってしまっているので修正
            def map_change_num(x):
                if x == '5 : だいたい予定が詰まっている (25%)':
                    x = '4 : だいたい予定が詰まっている (25%)'
                elif x == '4 : 朝から外せない予定がある (0%)':
                    x = '5 : 朝から外せない予定がある (0%)'
                return x
            self.df_answer.iloc[:, 17] = self.df_answer.iloc[:, 17].map(map_change_num)
            # 複数回答の区切り文字が";"と", "で２種類あるので", "に統一
            self.df_answer.iloc[:, 6] = self.df_answer.iloc[:, 6].map(lambda x: x.replace(';', ', ') if type(x) is str else x)
            # 遊園地と回答した結果を削除
            self.df_answer = self.df_answer.drop(self.df_answer.index[
                (self.df_answer.iloc[:, 3] == '遊園地・動物園・水族館・海水浴場') | 
                (self.df_answer.iloc[:, 4] == '遊園地・動物園・水族館・海水浴場') |
                (self.df_answer.iloc[:, 5] == '遊園地・動物園・水族館・海水浴場') |
                (self.df_answer.iloc[:, 18] == '遊園地・動物園・水族館・海水浴場')
            ]).reset_index(drop=True)
            # 複数回答の中に遊園地が含まれていたら行ごと削除（nanを"なし"に変換）
            self.df_answer.iloc[:, 6] = self.df_answer.iloc[:, 6].fillna('なし').map(lambda x: x.split(', '))
            # 複数回答の中に遊園地が含まれていたら行ごと削除
            self.df_answer = self.df_answer.drop(index=[i for i, s in enumerate(self.df_answer.iloc[:, 6]) if '遊園地・動物園・水族館・海水浴場' in s])
        preprocessing()
        
        # 各種データセット
        self.df_static_info_binary = None
        self.df_static_info_weight = None
        self.df_dynamic_info = None
        self.df_human_info = None
        
    def generate(self):
        # 回答結果を数値情報に変換
        df_answer_binary = self._convert_stoi(self.df_answer.copy())
        
        # 性格情報の学習データセットを作成
        self.df_human_info = self._generate_human_info(df_answer_binary)
        # 動的情報を扱うNNの学習データセットを作成
        self.df_dynamic_info = self._generate_dynamic_info(df_answer_binary)
        # 静的情報を扱うNNの学習データセットを作成
        self.df_static_info_binary, self.df_static_info_weight = self._generate_static_info(df_answer_binary)
        
    def _convert_stoi(self, df_answer):
        """
        回答結果を数値情報に変換するメソッド
        
        Parameters
        ----------
        df_answer : DataFrame
            回答結果
            
        Returns
        -------
        df_answer : DataFrame
            数値情報に変換した回答結果
        """
        
        # 性別を数値に変換
        genders = pd.factorize(df_answer.iloc[:, 1])
        df_answer.iloc[:, 1] = genders[0]
        
        # 遊びのジャンル（単回答）を数値に変換
        for i in [3, 4, 5, 18]:
            df_answer.iloc[:, i] = df_answer.iloc[:, i].fillna('なし').map(lambda x: self.GENRE_LIST.index(x))
            
        # 遊びのジャンル（複数回答）を数値に変換
        def map_stoi(lst_ans):
            lst_ans_int = []
            for ans in lst_ans:
                lst_ans_int.append(self.GENRE_LIST.index(ans))
            return lst_ans_int
        df_answer.iloc[:, 6] = df_answer.iloc[:, 6].map(map_stoi)
        
        # この時点で未回答の質問があるのはアウトなので除外
        df_answer = df_answer.dropna()
        # 連番付きの質問の回答を数値に変換
        for i in [2] + list(range(7, 18)) + list(range(19, len(df_answer.columns))):
            df_answer.iloc[:, i] = df_answer.iloc[:, i].map(lambda x: int(x[0]) - 1)
            
        # 数値情報に変換した回答結果を返す
        return df_answer
    
    def _generate_human_info(self, df_ans_bin):
        """
        回答結果から性格クラスタリングの学習データセットを作成するメソッド
        
        Parameters
        ----------
        df_ans_bin : DataFrame
            数値情報に変換した回答結果
            
        Returns
        -------
        df_ans_bin.iloc[:, 19:25] : DataFrame
            性格クラスタリングの学習データセット
        """
        return df_ans_bin.iloc[:, 19:25]
    
    def _generate_dynamic_info(self, df_ans_bin):
        """
        回答結果から動的情報を扱うNNの学習データセットを作成するメソッド
        
        Parameters
        ----------
        df_ans_bin : DataFrame
            数値情報に変換した回答結果
            
        Returns
        -------
        df_ans_bin.iloc[:, 19:25] : DataFrame
            動的情報を扱うNNの学習データセット
        """
        
        # 回答結果から必要な列を抽出
        df_dataset = df_ans_bin.iloc[:, 15:19]
        
        # 正解ラベルをベクトルに変換
        def map_label2vec(x):
            vector = np.zeros(len(self.GENRE_LIST) - 1).astype(np.int64)
            vector[x] = 1
            return vector
        df_dataset.iloc[:, -1] = df_dataset.iloc[:, -1].map(map_label2vec)
        
        # 生成したデータセットを返す
        return df_dataset
    
    def _generate_static_info(self, df_ans_bin):
        
        # 入力データと正解ラベルを分割
        def _split_input_label(df_ans):
            return df_ans.iloc[:, 4:], df_ans.iloc[:, :4]
        
        df_input, df_label = _split_input_label(df_ans_bin.iloc[:, 3:15])
        
        # バイナリパターンの学習データセットを作成
        def _generate_binary_label(df_in, df_lb):
            
            # 正解ラベルから第1~3希望の部分を抽出
            df_lb_bin = df_lb.iloc[:, :3]
            
            # 正解ラベルをバイナリのベクトルに変換
            def map_label2vec(row):
                vector = np.zeros(len(self.GENRE_LIST) - 1).astype(np.int64)
                for x in row:
                    if x != len(self.GENRE_LIST) - 1:
                        vector[x] = 1
                return vector
            df_in['label'] = df_lb_bin.apply(map_label2vec, axis=1)
            return df_in
        
        # 重み付きパターンの学習データセットを作成
        def _generate_weight_label(df_in, df_lb):
            
            # 複数回答した中に第1~3希望ですでに回答したものが含まれていたら除去する
            def apply_rm_duplicate(row):
                for i in range(3):
                    # 途中でリストが空になったらNoneを返す
                    if not row[3]:
                        return None
                    # 第1~3希望で選んだジャンルがリストに含まれていたら削除
                    if row[i] in row[3]:
                        row[3] = row[3].remove(row[i])

                # リストが空or"なし"のみの場合Noneを返す
                if not row[3] or (len(row[3]) == 1 and row[3][0] == len(self.GENRE_LIST) - 1):
                    return None
                
                return row[3]
            
            df_lb.iloc[:, 3] = df_lb.apply(apply_rm_duplicate, axis=1)
            
            # なしをNoneに変換
            df_lb.iloc[:, 1] = df_lb.iloc[:, 1].map(lambda x: None if x == len(self.GENRE_LIST) - 1 else x)
            df_lb.iloc[:, 2] = df_lb.iloc[:, 2].map(lambda x: None if x == len(self.GENRE_LIST) - 1 else x)
            
            # 正解ラベルを重み付きベクトルに変換
            def map_label2vec(row):
                weight = [1.0, 0.7, 0.4, 0.1]
                vector = np.zeros(len(self.GENRE_LIST) - 1).astype(np.float64)
                # 第1~3希望まで重みを計算
                for i, x in enumerate(row[:3]):
                    if not np.isnan(x):
                        vector[int(x)] = weight[i]
                # 第4希望の重みを計算
                if row[3] is not None:
                    for x in row[3]:
                        vector[int(x)] = weight[-1]
                return vector
            
            df_in['label'] = df_lb.apply(map_label2vec, axis=1)
            return df_in
        
        # バイナリベクトルと重み付きベクトルを返す
        return _generate_binary_label(df_input.copy(), df_label.copy()), _generate_weight_label(df_input.copy(), df_label.copy())
