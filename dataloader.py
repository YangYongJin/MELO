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
    def __init__(self, file_path, max_sequence_length=10, min_sequence=5, min_window_size=2, samples_per_task=25, num_test_data=500,  mode="ml-1m", default_rating=0, pretraining=False, pretraining_batch_size=None):
        '''
        Args:
            file_path : path of file containing target data
            max_sequence_length : maximum sequence length to sample
            min_sequence : minimum sequence used to filter users
            min_window_size : minimum window size during subsampling
            samples_per_task : number of subsamples for each user
            mode : "amazon" or "ml-1m"
            default_rating : padding options
            pretraining : when used for pretraining or single bert model
            pretraining_batch_size : batch size during pretraining
        '''

        # make data directory
        os.makedirs(os.path.join(os.path.abspath(
            '.'), ROOT_FOLDER), exist_ok=True)
        self.max_sequence_length = max_sequence_length
        self.min_window_size = min_window_size

        self.df, self.umap, self.smap = self.preprocessing(
            file_path, min_sequence, mode)
        self.num_samples = samples_per_task
        self.train_set, self.valid_set, self.test_set = self.split_data(
            self.df, num_test_data)
        self.num_items = len(self.smap)
        self.num_users = len(self.umap)
        self.total_data_num = len(self.df)

        self.default_rating = default_rating

        # for pretraining (learn sigle bert)
        if pretraining and pretraining_batch_size != None:
            self.pretraining_train_loader = self.make_pretraining_dataloader(
                self.train_set, pretraining_batch_size)
            self.pretraining_valid_loader = self.make_pretraining_dataloader(
                self.valid_set, pretraining_batch_size)
            self.pretraining_test_loader = self.make_pretraining_dataloader(
                self.test_set, pretraining_batch_size)

    def preprocessing(self, file_path, min_sequence, mode="ml-1m"):
        '''
        Preprocessing data
        return data with user - sequence pairs

        Args:
            file_path : path of file containing target data
            min_sequence : minimum sequence used to filter users
            mode : "amazon" or "ml-1m"

        return:
            df : preprocessed data
            umap : user ids
            smap : product ids
        '''
        print("Preprocessing Started")
        if mode == "ml-1m":
            self.download_raw_movielnes_data()
            raw_df = pd.read_csv(file_path, sep='::',
                                 header=None, engine="python")
            raw_df.columns = ['user_id', 'product_id', 'rating', 'date']
        elif mode == "amazon":
            # choose appropriate columns
            raw_df = pd.read_csv(file_path, usecols=[
                'rating', 'reviewerID', 'product_id', 'date'])
            raw_df = raw_df.iloc[:500000, :]
            raw_df.rename(columns={'reviewerID': 'user_id'}, inplace=True)
            raw_df.loc[:, 'rating'] = raw_df.loc[:,
                                                 'rating'].apply(lambda x: float(x))

        # filter user with lack of reviews
        raw_df = self.filter_triplets(raw_df, min_sequence)

        # map user or product id => int
        raw_df, umap, smap = self.densify_index(raw_df)

        # sort by data => make sequence
        df_group = raw_df.sort_values(by=['date']).groupby('user_id')

        df = pd.DataFrame(
            data={
                'user_id': list(df_group.groups.keys()),
                'product_id': list(df_group.product_id.apply(list)),
                'rating': list(df_group.rating.apply(list)),
                'date': list(df_group.date.apply(list)),
            }
        )

        del df['date']
        print("Preprocessing Finished!")
        return df, umap, smap

    def download_raw_movielnes_data(self):
        '''
            This function downloads movielens-1m
        '''
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
        '''
        filter user with lack of reviews
        Args:
            df: input data
            min_sequence: minimum reviews users should have
        return:
            df : filtered data
        '''
        item_sizes = df.groupby('product_id').size()
        good_items = item_sizes.index[item_sizes >= 2]
        df = df[df['product_id'].isin(good_items)]

        user_sizes = df.groupby('user_id').size()
        good_users = user_sizes.index[user_sizes >= min_sequence]
        df = df[df['user_id'].isin(good_users)]

        return df

    def densify_index(self, df):
        '''
        densify index - map id => int number with range(0, num_ids)
        Args:
            df: input data
        return:
            df : densified data
            umap : user ids
            smap : product ids
        '''
        umap = {u: i for i, u in enumerate(set(df['user_id']))}
        smap = {s: i for i, s in enumerate(set(df['product_id']))}
        df['user_id'] = df['user_id'].map(umap)
        df['product_id'] = df['product_id'].map(smap)
        return df, umap, smap

    def split_data(self, df, num_test_data=500):
        '''
            split train, test, valid
        '''
        test_data = df[-num_test_data:]
        used_df = df[:-num_test_data]
        random_selection = np.random.rand(len(used_df.index)) <= 0.85
        train_data = used_df[random_selection]
        valid_data = used_df[~random_selection]
        return train_data, valid_data, test_data

    def cut_sequences(self, values, rand_idx):
        '''
        cut sequences
        Args:
            values : seqeunce
            rand_idx : start index to cut
        return:
            cut sequence
        '''
        if len(values) <= self.max_sequence_length:
            return values[:]
        else:
            return values[rand_idx: rand_idx+self.max_sequence_length]

    def get_sliced_sequences(self, product_ids, ratings):
        '''
            cut product_ids and ratings
        '''
        cut_num = len(product_ids) if len(
            product_ids) < self.max_sequence_length else self.max_sequence_length
        rand_idx = np.random.randint(
            len(product_ids) + 1 - cut_num)
        ratings = self.cut_sequences(ratings, rand_idx)
        product_ids = self.cut_sequences(product_ids, rand_idx)
        return ratings, product_ids

    def subsample(self, values, rand_idxs):
        '''
        subsampling function
        Args:
            values : sequence
            rand_idx : random indexs to sample from sequence cut
        return:
            subsampled sequences
        '''
        sequences = []
        max_window_size = len(values)
        for window_size in range(self.min_window_size, max_window_size+1):
            for start_idx in range(max_window_size-window_size+1):
                sequence = [0] * (self.max_sequence_length - window_size) + \
                    list(values[start_idx: start_idx+window_size])
                sequences.append(sequence)
        sequences = np.asarray(sequences)
        return sequences[rand_idxs]

    def preprocess_wt_subsampling(self, product_ids, ratings):
        '''
        subsampling geneartion pipieline function
        cut sequence => subsample => make torch tensor
        Args:
            product_ids : product_ids
            ratings : ratings
        return:
            product_ids : subsampled product_ids
            ratings : subsampled ratings
            normalized_num_samples: # samples / expected # samples
            target_idx : target product(rating) index
        '''
        ratings, product_ids = self.get_sliced_sequences(product_ids, ratings)
        # number of subsamples (1+2+ ... + n-min_window+1 = (n-min_window+2)*(n-min_window+1)/2)
        cur_num_samples = (len(ratings)-self.min_window_size+2) * \
            (len(ratings)-self.min_window_size+1)//2
        num_subsamples = cur_num_samples if cur_num_samples < self.num_samples else self.num_samples
        rand_idxs = np.random.choice(
            cur_num_samples, num_subsamples, replace=False)
        ratings = torch.FloatTensor(self.subsample(ratings, rand_idxs))
        product_ids = torch.LongTensor(self.subsample(product_ids, rand_idxs))
        normalized_num_samples = num_subsamples/self.num_samples

        # choose target index
        target_idx = np.random.randint(num_subsamples)
        return ratings, product_ids, normalized_num_samples, target_idx

    def make_support_set(self, user_id, product_ids, ratings, target_idx, normalized=False):
        '''
            function that makes support set
            choose all except target index element
        '''
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

        # make rating information based on supper ratings
        rating_info = self.make_rating_info(
            support_rating_history, normalized)

        # set default rating for padding
        support_rating_history = support_rating_history + \
            self.default_rating*(support_rating_history == 0)

        support_data = (support_user_id, support_product_history,
                        support_target_product, support_rating_history, support_target_rating)
        return support_data, rating_info

    def make_query_set(self, user_id, product_ids, ratings, target_idx):
        '''
            function that makes query set
            choose target index element
        '''
        query_product_history = product_ids[target_idx:target_idx+1, :-1]
        query_target_product = product_ids[target_idx:target_idx+1, -1:]
        query_rating_history = ratings[target_idx:target_idx+1, :-1]
        query_target_rating = ratings[target_idx:target_idx+1, -1:]
        query_user_id = user_id.repeat(
            1, 1)

        # set default rating for padding
        query_rating_history = query_rating_history + \
            self.default_rating*(query_rating_history == 0)

        query_data = (query_user_id, query_product_history,
                      query_target_product, query_rating_history, query_target_rating)
        return query_data

    def make_rating_info(self, support_rating_history, normalized=False):
        '''
        function that makes task information about rating
        normalization option : rating with range(0,1)

        return:
            num_i : # of ratings with i value / # of total ratings with all values
            rating_mean: mean value of ratings
            rating_std: std value of ratings
        '''
        total_ratings = (support_rating_history > 0).sum()
        num_1 = (support_rating_history == 1).sum()/total_ratings
        num_2 = (support_rating_history == 2).sum()/total_ratings
        num_3 = (support_rating_history == 3).sum()/total_ratings
        num_4 = (support_rating_history == 4).sum()/total_ratings
        num_5 = (support_rating_history == 5).sum()/total_ratings

        if normalized:
            rating_mean = (support_rating_history/5.0).mean()
            rating_std = (support_rating_history/5.0).std()
        else:
            rating_mean = support_rating_history.mean()
            rating_std = support_rating_history.std()
        return [rating_mean, rating_std, num_1, num_2, num_3, num_4, num_5]

    def generate_task(self, mode="train", batch_size=20, normalized=False):
        '''
        generate batch of tasks

        Args:
            mode : train or valid
            batch_size : task batch size
            normalized : use normalized version of ratings
        return:
            tasks : batch of (support_set, query_set, task_info)
        '''
        if mode == "train":
            data_set = self.train_set
        elif mode == "valid":
            data_set = self.valid_set
        elif mode == "test":
            data_set = self.test_set
        tasks = []
        idxs = np.random.choice(len(data_set.index),
                                batch_size, replace=False)
        for i in idxs:
            data = data_set.iloc[i]
            user_id = torch.tensor(data.user_id)
            product_ids = data.product_id
            ratings = data.rating

            # subsamples
            ratings, product_ids, normalized_num_samples, target_idx = self.preprocess_wt_subsampling(
                product_ids, ratings)

            # make support set and query set
            support_data, rating_info = self.make_support_set(
                user_id, product_ids, ratings, target_idx, normalized)

            query_data = self.make_query_set(
                user_id, product_ids, ratings, target_idx)

            # make task information
            task_info = torch.Tensor(rating_info + [normalized_num_samples])

            tasks.append((support_data, query_data, task_info))

        return tasks

    def make_pretraining_dataloader(self, df, batch_size=128):
        '''
        funtion that makes dataloader for pretraining(single bert model)
        Args:
            df: data(train or valid)
            batch_size: training batch size
        return:
            dataloader: torch dataloader
        '''
        dataset = SequenceDataset(
            df, self.max_sequence_length, self.min_window_size, self.default_rating)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return dataloader


class SequenceDataset(data.Dataset):
    """
        Pytorch dataset for review data
    """

    def __init__(
        self, df, max_len, min_window_size=2, default_rating=0
    ):
        """
        Args:
            df: preprocessed data
            max_len: max sequence length
            min_window_size : minimum window size during subsampling
            default_rating: rating of padding
        """
        self.ratings_frame = df
        self.max_len = max_len
        self.min_window_size = min_window_size
        self.default_rating = default_rating

    def __len__(self):
        return len(self.ratings_frame)

    def preprocessing(self, product_ids, ratings):
        """
            function that makes single random sequence for each user
            sequence length ranges from min_window_size to max_len
        """
        seq_len = len(product_ids)
        maximum_len = seq_len if seq_len < self.max_len else self.max_len
        window_size = np.random.randint(self.min_window_size, maximum_len+1)
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

        # if we want default ratings 1
        ratings = ratings + self.default_rating*(ratings == 0)

        product_history = product_ids[:-1]
        target_product_id = product_ids[-1:][0].reshape(1)
        product_history_ratings = ratings[:-1]
        target_product_rating = ratings[-1:][0].reshape(1)

        return (user_id, product_history, target_product_id,  product_history_ratings), target_product_rating


# dataloader = DataLoader('./Data/ml-1m/ratings.dat', max_sequence_length=30, min_sequence=5, min_window_size=15,
#                         samples_per_task=64, num_test_data=500,  mode="ml-1m", default_rating=1, pretraining=False, pretraining_batch_size=None)
# tasks = dataloader.generate_task(mode="train", batch_size=25, normalized=True)
# support, query, task = tasks[0]
# support_user_id, support_product_history, support_target_product, support_rating_history, support_target_rating = support
# print(((support_product_history)==0).sum(axis=1))
