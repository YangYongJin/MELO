import numpy as np
import torch
import pandas as pd
import shutil
import tempfile
import os
import wget
import zipfile
from pathlib import Path


class DataLoader():
    def __init__(self, file_path, mode="ml-1m"):
        self.sequence_length = 5
        self.df, self.umap, self.smap = self.preprocessing(file_path, mode)
        self.train_set, self.valid_set = self.split_data(self.df)
        self.num_items = len(self.smap)
        self.num_users = len(self.umap)
        self.total_data_num = len(self.df)

    def preprocessing(self, file_path, mode="ml-1m"):
        print("Preprocessing Started")
        if mode == "ml-1m":
            self.download_raw_movielnes_data()
            raw_df = pd.read_csv(file_path, sep='::',
                                 header=None, engine="python")
            raw_df.columns = ['user_id', 'product_id', 'rating', 'date']
        elif mode == "amazon":
            raw_df = pd.read_csv(file_path, usecols=[
                'rating', 'reviewerID', 'product_id', 'date'])
            raw_df = raw_df.iloc[:500000, :]
            raw_df.rename(columns={'reviewerID': 'user_id'}, inplace=True)
            raw_df.loc[:, 'rating'] = raw_df.loc[:,
                                                 'rating'].apply(lambda x: float(x))

        raw_df = self.filter_triplets(raw_df)

        raw_df, umap, smap = self.densify_index(raw_df)

        df_group = raw_df.sort_values(by=['date']).groupby('user_id')

        df = pd.DataFrame(
            data={
                'user_id': list(df_group.groups.keys()),
                'product_id': list(df_group.product_id.apply(list)),
                'rating': list(df_group.rating.apply(list)),
                'date': list(df_group.date.apply(list)),
            }
        )
        step_size = 1
        df.product_id = df.product_id.apply(
            lambda ids: self.create_sequences(
                ids, self.sequence_length, step_size)
        )

        df.rating = df.rating.apply(
            lambda ids: self.create_sequences(
                ids, self.sequence_length, step_size)
        )

        del df['date']
        print("Preprocessing Finished!")
        return df, umap, smap

    def download_raw_movielnes_data(self):
        folder_path = Path("Data").joinpath("ml-1m")
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        if True:
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            wget.download(self.get_url(), str(tmpzip))
            zip = zipfile.ZipFile(tmpzip)
            zip.extractall(tmpfolder)
            zip.close()
            tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()

    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'users.dat']

    def get_url(self):
        return "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def filter_triplets(self, df):
        item_sizes = df.groupby('product_id').size()
        good_items = item_sizes.index[item_sizes >= 2]
        df = df[df['product_id'].isin(good_items)]

        user_sizes = df.groupby('user_id').size()
        good_users = user_sizes.index[user_sizes >= 5]
        df = df[df['user_id'].isin(good_users)]

        return df

    def densify_index(self, df):
        umap = {u: i for i, u in enumerate(set(df['user_id']))}
        smap = {s: i for i, s in enumerate(set(df['product_id']))}
        df['user_id'] = df['user_id'].map(umap)
        df['product_id'] = df['product_id'].map(smap)
        return df, umap, smap

    def split_data(self, df):
        random_selection = np.random.rand(len(df.index)) <= 0.85
        train_data = df[random_selection]
        test_data = df[~random_selection]
        return train_data, test_data

    def create_sequences(self, values, window_size, step_size):
        sequences = []
        start_index = 0
        while True:
            end_index = start_index + window_size
            seq = values[start_index:end_index]
            if len(seq) < window_size:
                seq = values[-window_size:]
                if len(seq) == window_size:
                    sequences.append(seq)
                break
            sequences.append(seq)
            start_index += step_size
        return sequences

    def generate_task(self, mode="train", batch_size=20, N=3, query_num=1):
        if mode == "train":
            data_set = self.train_set
        elif mode == "valid":
            data_set = self.valid_set
        tasks = []
        idxs = np.random.choice(len(data_set.index),
                                batch_size*N, replace=False)
        idxs = idxs.reshape(batch_size, N)
        for batch in idxs:
            support_target_products = []
            support_rating_historys = []
            support_target_ratings = []
            support_product_historys = []
            support_user_ids = []

            query_target_products = []
            query_rating_historys = []
            query_target_ratings = []
            query_product_historys = []
            query_user_ids = []

            for i in batch:
                data = data_set.iloc[i]
                user_id = torch.tensor(data.user_id)
                product_ids = torch.LongTensor(data.product_id)
                ratings = torch.FloatTensor(data.rating)
                support_product_historys.append(product_ids[:-query_num, :-1])
                support_target_products.append(product_ids[:-query_num, -1:])
                support_rating_historys.append(ratings[:-query_num, :-1])
                support_target_ratings.append(ratings[:-query_num, -1:])
                support_user_ids.append(user_id.repeat(
                    len(ratings)-query_num, 1))

                query_product_historys.append(product_ids[-query_num:, :-1])
                query_target_products.append(product_ids[-query_num:, -1:])
                query_rating_historys.append(ratings[-query_num:, :-1])
                query_target_ratings.append(ratings[-query_num:, -1:])
                query_user_ids.append(user_id.repeat(query_num, 1))

            support_target_product = torch.cat(support_target_products, dim=0)
            support_rating_history = torch.cat(support_rating_historys, dim=0)
            support_target_rating = torch.cat(support_target_ratings, dim=0)
            support_product_history = torch.cat(
                support_product_historys, dim=0)
            support_user_id = torch.cat(support_user_ids, dim=0)

            query_target_product = torch.cat(query_target_products, dim=0)
            query_rating_history = torch.cat(query_rating_historys, dim=0)
            query_target_rating = torch.cat(query_target_ratings, dim=0)
            query_product_history = torch.cat(query_product_historys, dim=0)
            query_user_id = torch.cat(query_user_ids, dim=0)

            support_data = (support_user_id, support_product_history,
                            support_target_product, support_rating_history, support_target_rating)
            query_data = (query_user_id, query_product_history,
                          query_target_product, query_rating_history, query_target_rating)

            tasks.append((support_data, query_data))

        return tasks


# dataloader = DataLoader('./Data/ml-1m/Office_Products.csv')
# tasks = dataloader.generate_task()
# print(tasks[0][0][0])
# print(tasks[5][0][0])
