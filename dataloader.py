import numpy as np
import torch
import pandas as pd
import shutil
import torch.utils.data as data
import tempfile
import os
import wget
import zipfile
from pathlib import Path

ROOT_FOLDER = "Data"


class DataLoader():
    def __init__(self, file_path, max_sequence_length=10, min_sequence=5, samples_per_task=25, mode="ml-1m", pretraining=False, pretraining_batch_size=None):
        os.makedirs(os.path.join(os.path.abspath(
            '.'), ROOT_FOLDER), exist_ok=True)
        self.max_sequence_length = max_sequence_length
        self.df, self.umap, self.smap = self.preprocessing(
            file_path, min_sequence, mode)
        self.num_samples = samples_per_task
        self.train_set, self.valid_set = self.split_data(self.df)
        self.num_items = len(self.smap)
        self.num_users = len(self.umap)
        self.total_data_num = len(self.df)

        # pretraining set
        if pretraining and pretraining_batch_size != None:
            self.pretraining_train_loader = self.make_pretraining_dataloader(
                self.train_set, pretraining_batch_size)
            self.pretraining_valid_loader = self.make_pretraining_dataloader(
                self.valid_set, pretraining_batch_size)

    def preprocessing(self, file_path, min_sequence, mode="ml-1m"):
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

        raw_df = self.filter_triplets(raw_df, min_sequence)

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

        # df.product_id = df.product_id.apply(
        #     lambda ids: self.cut_sequences(
        #         ids)
        # )

        # df.rating = df.rating.apply(
        #     lambda ids: self.cut_sequences(
        #         ids)
        # )

        # df.product_id = df.product_id.apply(
        #     lambda ids: self.subsample(
        #         ids)
        # )

        # df.rating = df.rating.apply(
        #     lambda ids: self.subsample(
        #         ids)

        del df['date']
        print("Preprocessing Finished!")
        return df, umap, smap

    def download_raw_movielnes_data(self):
        folder_path = Path(ROOT_FOLDER).joinpath("ml-1m")
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

    def filter_triplets(self, df, min_sequence):
        item_sizes = df.groupby('product_id').size()
        good_items = item_sizes.index[item_sizes >= 2]
        df = df[df['product_id'].isin(good_items)]

        user_sizes = df.groupby('user_id').size()
        good_users = user_sizes.index[user_sizes >= min_sequence]
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

    def cut_sequences(self, values, rand_idx):
        if len(values) <= self.max_sequence_length:
            return values[:]
        else:
            return values[rand_idx: rand_idx+self.max_sequence_length]

    def get_sliced_sequences(self, product_ids, ratings):
        rand_idx = np.random.randint(
            len(product_ids) + 1 - self.max_sequence_length)
        ratings = self.cut_sequences(ratings, rand_idx)
        product_ids = self.cut_sequences(product_ids, rand_idx)
        return ratings, product_ids

    def subsample(self, values, rand_idxs):
        sequences = []
        max_window_size = len(values)
        for window_size in range(2, max_window_size+1):
            for start_idx in range(max_window_size-window_size+1):
                sequence = [0] * (self.max_sequence_length - window_size) + \
                    list(values[start_idx: start_idx+window_size])
                sequences.append(sequence)
        sequences = np.asarray(sequences)
        return sequences[rand_idxs]

    def preprocess_wt_subsampling(self, product_ids, ratings):
        ratings, product_ids = self.get_sliced_sequences(product_ids, ratings)
        cur_num_samples = len(ratings)*(len(ratings)-1)//2
        num_subsamples = cur_num_samples if cur_num_samples < self.num_samples else self.num_samples
        rand_idxs = np.random.choice(
            cur_num_samples, num_subsamples, replace=False)
        ratings = torch.FloatTensor(self.subsample(ratings, rand_idxs))
        product_ids = torch.LongTensor(self.subsample(product_ids, rand_idxs))
        normalized_num_samples = num_subsamples/self.num_samples
        target_idx = np.random.randint(num_subsamples)
        return ratings, product_ids, normalized_num_samples, target_idx

    def make_support_set(self, user_id, product_ids, ratings, target_idx):
        if target_idx >= (len(ratings)-1):
            support_idxs = torch.LongTensor(np.arange(target_idx))
        else:
            support_idxs = torch.cat((torch.LongTensor(np.arange(target_idx)), torch.LongTensor(
                np.arange(target_idx+1, len(product_ids)))))
        support_product_history = product_ids[support_idxs, :-1]
        support_target_product = product_ids[support_idxs, -1:]
        support_rating_history = ratings[support_idxs, :-1]
        support_target_rating = ratings[support_idxs, -1:]
        support_user_id = user_id.repeat(
            len(ratings)-1, 1)

        support_data = (support_user_id, support_product_history,
                        support_target_product, support_rating_history, support_target_rating)
        return support_data, support_target_rating

    def make_query_set(self, user_id, product_ids, ratings, target_idx):
        query_product_history = product_ids[target_idx:target_idx+1, :-1]
        query_target_product = product_ids[target_idx:target_idx+1, -1:]
        query_rating_history = ratings[target_idx:target_idx+1, :-1]
        query_target_rating = ratings[target_idx:target_idx+1, -1:]
        query_user_id = user_id.repeat(
            1, 1)

        query_data = (query_user_id, query_product_history,
                      query_target_product, query_rating_history, query_target_rating)
        return query_data

    def make_rating_info(self, support_target_rating):
        num_1 = (support_target_rating == 1).sum()/len(support_target_rating)
        num_2 = (support_target_rating == 2).sum()/len(support_target_rating)
        num_3 = (support_target_rating == 3).sum()/len(support_target_rating)
        num_4 = (support_target_rating == 4).sum()/len(support_target_rating)
        num_5 = (support_target_rating == 5).sum()/len(support_target_rating)
        rating_mean = support_target_rating.mean()
        rating_std = support_target_rating.std()
        return [rating_mean, rating_std, num_1, num_2, num_3, num_4, num_5]

    def generate_task(self, mode="train", batch_size=20):
        if mode == "train":
            data_set = self.train_set
        elif mode == "valid":
            data_set = self.valid_set
        tasks = []
        idxs = np.random.choice(len(data_set.index),
                                batch_size, replace=False)
        for i in idxs:
            data = data_set.iloc[i]
            user_id = torch.tensor(data.user_id)
            product_ids = data.product_id
            ratings = data.rating
            ratings, product_ids, normalized_num_samples, target_idx = self.preprocess_wt_subsampling(
                product_ids, ratings)

            support_data, support_target_rating = self.make_support_set(
                user_id, product_ids, ratings, target_idx)

            rating_info = self.make_rating_info(support_target_rating)

            query_data = self.make_query_set(
                user_id, product_ids, ratings, target_idx)

            task_info = torch.Tensor(rating_info + [normalized_num_samples])

            tasks.append((support_data, query_data, task_info))

        return tasks

    def make_pretraining_dataloader(self, df, batch_size=128):
        dataset = SequenceDataset(df, self.max_sequence_length)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return dataloader


class SequenceDataset(data.Dataset):
    """Movie dataset."""

    def __init__(
        self, df, max_len
    ):
        """
        Args:
            csv_file (string): Path to the csv file with user,past,future.
        """
        self.ratings_frame = df
        self.max_len = max_len

    def __len__(self):
        return len(self.ratings_frame)

    def preprocessing(self, product_ids, ratings):
        seq_len = len(product_ids)
        window_size = np.random.randint(2, self.max_len+1)
        start_idx = np.random.randint(0, seq_len - window_size + 1)
        product_ids_f = [0] * (self.max_len - window_size) + \
            list(product_ids[start_idx: start_idx+window_size])
        ratings_f = [0] * (self.max_len - window_size) + \
            list(ratings[start_idx: start_idx+window_size])
        product_ids_f = torch.LongTensor(product_ids_f)
        ratings_f = torch.FloatTensor(ratings_f)
        return product_ids_f, ratings_f

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = torch.tensor(data.user_id).reshape(1)
        product_ids = data.product_id
        ratings = data.rating

        product_ids, ratings = self.preprocessing(
            product_ids, ratings)

        product_history = product_ids[:-1]
        target_product_id = product_ids[-1:][0].reshape(1)
        product_history_ratings = ratings[:-1]
        target_product_rating = ratings[-1:][0].reshape(1)

        return (user_id, product_history, target_product_id,  product_history_ratings), target_product_rating

# dataloader = DataLoader('./Data/ml-1m/ratings.dat')
# tasks = dataloader.generate_task()
# support, query, task_info = tasks[0]
# print(support[1].shape)
# print(query[1].shape)
# print(task_info)
