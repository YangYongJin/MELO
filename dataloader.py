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
from options import args

ROOT_FOLDER = "Data"


class DataLoader():
    def __init__(self, args, pretraining):
        '''
        Args:
            data_path : path of file containing target data
            max_sequence_length : maximum sequence length to sample
            min_sequence : minimum sequence used to filter users
            min_item : minimum item size used to filter items
            min_sub_window_size : minimum window size during subsampling
            samples_per_task : number of subsamples for each user
            num_test_data : the number of test data
            random_seed : random seed for test data sampling
            mode : "amazon" or "ml-1m" or "yelp"
            default_rating : padding options
            pretraining : when used for pretraining or single bert model
            pretraining_batch_size : batch size during pretraining
        '''

        # make data directory
        os.makedirs(os.path.join(os.path.abspath(
            '.'), ROOT_FOLDER), exist_ok=True)
        self.max_sequence_length = args.max_seq_len
        self.min_sub_window_size = args.min_sub_window_size

        self.random_seed = args.random_seed
        self.num_query_set = args.num_query_set

        self.df, self.umap, self.smap = self.preprocessing(
            args.data_path, args.min_sequence, args.min_item, args.mode)
        self.num_samples = args.num_samples
        self.train_set, self.valid_set, self.test_set = self.split_data(
            self.df, args.num_test_data)
        self.num_items = len(self.smap)
        self.num_users = len(self.umap)
        self.total_data_num = len(self.df)

        self.default_rating = args.default_rating

        self.batch_idxs = []
        self.batch_idx = 0
        # task information:
        self.task_info_rating_mean = args.task_info_rating_mean
        self.task_info_rating_std = args.task_info_rating_std
        self.task_info_num_samples = args.task_info_num_samples
        self.task_info_rating_distribution = args.task_info_rating_distribution

        # for pretraining (learn sigle bert)
        if pretraining and args.pretraining_batch_size != None:
            self.pretraining_train_loader = self.make_pretraining_dataloader(
                self.train_set, args.pretraining_batch_size)
            self.pretraining_valid_loader = self.make_pretraining_dataloader(
                self.valid_set, args.pretraining_batch_size)
            self.pretraining_test_loader = self.make_pretraining_dataloader(
                self.test_set, args.pretraining_batch_size, num_queries=args.num_query_set)

    def preprocessing(self, data_path, min_sequence, min_item, mode="ml-1m"):
        '''
        Preprocessing data
        return data with user - sequence pairs

        Args:
            data_path : path of file containing target data
            min_sequence : minimum sequence used to filter users
            min_item : minimum item size used to filter items
            mode : "amazon" or "ml-1m" or "yelp"

        return:
            df : preprocessed data
            umap : user ids
            smap : product ids
        '''
        print("Preprocessing Started")
        if mode == "ml-1m":
            self.download_raw_movielnes_data()
            raw_df = pd.read_csv(data_path, sep='::',
                                 header=None, engine="python")
            raw_df.columns = ['user_id', 'product_id', 'rating', 'date']
        elif mode == "amazon":
            # choose appropriate columns
            raw_df = pd.read_csv(data_path, usecols=[
                'rating', 'reviewerID', 'product_id', 'date'])
            raw_df.rename(columns={'reviewerID': 'user_id'}, inplace=True)
        elif mode == "yelp":
            # choose appropriate columns
            raw_df = pd.read_csv(data_path, usecols=[
                'stars', 'user_id', 'business_id', 'timestamp'])
            raw_df.rename(columns={'business_id': 'product_id'}, inplace=True)
            raw_df.rename(columns={'timestamp': 'date'}, inplace=True)

        # filter user with lack of reviews
        raw_df = self.filter_triplets(raw_df, min_sequence, min_item)

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

    def filter_triplets(self, df, min_sequence, min_item):
        '''
        filter user with lack of reviews
        Args:
            df: input data
            min_sequence: minimum reviews users should have
            min_item: minimum reviews items should have
        return:
            df : filtered data
        '''
        item_sizes = df.groupby('product_id').size()
        good_items = item_sizes.index[item_sizes >= min_item]
        df = df[df['product_id'].isin(good_items)]

        user_sizes = df.groupby('user_id').size()
        good_users = user_sizes.index[user_sizes >= min_sequence]
        df = df[df['user_id'].isin(good_users)]

        print("total number of data", len(df))

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
        smap = {s: (i+1) for i, s in enumerate(set(df['product_id']))}
        df['user_id'] = df['user_id'].map(umap)
        df['product_id'] = df['product_id'].map(smap)
        return df, umap, smap

    def split_data(self, df, num_test_data=500):
        '''
            split train, test, valid
        '''
        np.random.seed(self.random_seed)
        test_idxs = np.random.choice(
            len(df.index), num_test_data, replace=False)
        train_valid_idxs = np.setdiff1d(range(len(df.index)), test_idxs)
        test_data = df.iloc[test_idxs]
        used_df = df.iloc[train_valid_idxs]
        np.random.seed()
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
        for window_size in range(self.min_sub_window_size, max_window_size+1):
            for start_idx in range(max_window_size-window_size+1):
                sequence = [0] * (self.max_sequence_length - window_size) + \
                    list(values[start_idx: start_idx+window_size])
                sequences.append(sequence)
        sequences = np.asarray(sequences)
        return sequences[rand_idxs]

    def make_query_seq(self, ratings, product_ids):
        query_ratings = []
        query_product_ids = []
        start_idxs = np.random.choice(
            len(ratings)-2, self.num_query_set, replace=False)
        for start_idx in start_idxs:
            ratings_adapt = ratings[start_idx:]
            product_ids_adapt = product_ids[start_idx:]
            query_rating = [0] * (self.max_sequence_length -
                                  len(ratings_adapt)) + ratings_adapt
            query_product_id = [0] * (self.max_sequence_length -
                                      len(product_ids_adapt)) + product_ids_adapt
            query_rating = torch.FloatTensor(query_rating)
            query_product_id = torch.LongTensor(query_product_id)

            query_ratings.append(query_rating)
            query_product_ids.append(query_product_id)
        query_ratings = torch.stack(query_ratings)
        query_product_ids = torch.stack(query_product_ids)
        return query_ratings, query_product_ids

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

        process:
            cut sequences -> get query -> subsample support
        '''
        ratings, product_ids = self.get_sliced_sequences(product_ids, ratings)

        query_ratings, query_product_ids = self.make_query_seq(
            ratings, product_ids)
        # number of support subsamples (1+2+ ... + (n-min_window) = (n-min_window+1)*(n-min_window)/2)
        cur_num_samples = (len(ratings)-self.min_sub_window_size+1) * \
            (len(ratings)-self.min_sub_window_size)//2
        num_subsamples = cur_num_samples if cur_num_samples < self.num_samples else self.num_samples
        rand_idxs = np.random.choice(
            cur_num_samples, num_subsamples, replace=False)
        support_ratings = torch.FloatTensor(
            self.subsample(ratings[:-1], rand_idxs))
        support_product_ids = torch.LongTensor(
            self.subsample(product_ids[:-1], rand_idxs))
        normalized_num_samples = num_subsamples/self.num_samples

        return support_ratings, support_product_ids, query_ratings, query_product_ids, normalized_num_samples

    def make_support_set(self, user_id, product_ids, ratings,  normalized=False, use_label=True):
        '''
            function that makes support set
            choose all except target index element

            Args:
                use_label : use label or not
        '''

        support_product_history = product_ids[:, :-1]
        support_target_product = product_ids[:, -1:]
        support_rating_history = ratings[:, :-1]
        support_target_rating = ratings[:, -1:]
        support_user_id = user_id.repeat(
            len(ratings), 1)

        # make rating information based on supper ratings
        if use_label:
            rating_info = self.make_rating_info(
                support_target_rating, normalized)
        else:
            rating_info = self.make_rating_info(
                support_rating_history, normalized)

        # set default rating for padding
        support_rating_history = support_rating_history + \
            self.default_rating*(support_rating_history == 0)

        support_data = (support_user_id, support_product_history,
                        support_target_product, support_rating_history, support_target_rating)
        return support_data, rating_info

    def make_query_set(self, user_id, product_ids, ratings):
        '''
            function that makes query set
            choose target index element
        '''

        query_product_history = product_ids[:, :-1]
        query_target_product = product_ids[:, -1:]
        query_rating_history = ratings[:, :-1]
        query_target_rating = ratings[:, -1:]
        query_user_id = user_id.repeat(
            len(product_ids), 1)

        # set default rating for padding
        query_rating_history = query_rating_history + \
            self.default_rating*(query_rating_history == 0)

        query_data = (query_user_id, query_product_history,
                      query_target_product, query_rating_history, query_target_rating)
        return query_data

    def make_rating_info(self, ratings, normalized=False):
        '''
        function that makes task information about rating
        normalization option : rating with range(0,1)

        return:
            num_i : # of ratings with i value / # of total ratings with all values
            rating_mean: mean value of ratings
            rating_std: std value of ratings
        '''
        total_ratings = (ratings > 0).sum()
        num_1 = int((ratings == 1).sum().item())
        num_2 = int((ratings == 2).sum().item())
        num_3 = int((ratings == 3).sum().item())
        num_4 = int((ratings == 4).sum().item())
        num_5 = int((ratings == 5).sum().item())

        rating_info = num_1 * [1] + num_2 * [2] + \
            num_3*[3] + num_4*[4] + num_5 * [5]

        normalized_num_1 = num_1 / total_ratings
        normalized_num_2 = num_2 / total_ratings
        normalized_num_3 = num_3 / total_ratings
        normalized_num_4 = num_4 / total_ratings
        normalized_num_5 = num_5 / total_ratings

        rating_info = torch.FloatTensor(rating_info)

        if normalized:
            rating_mean = (rating_info/5.0).mean()
            rating_std = (rating_info/5.0).std()
        else:
            rating_mean = rating_info.mean()
            rating_std = rating_info.std()
        rating_info = self.task_info_rating_mean*[rating_mean] + self.task_info_rating_std*[rating_std] + self.task_info_rating_distribution * [
            normalized_num_1, normalized_num_2, normalized_num_3, normalized_num_4, normalized_num_5]

        return rating_info

    def generate_task(self, mode="train", batch_size=20, normalized=False, use_label=True):
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

        if mode == "train":
            if len(self.batch_idxs) == 0:
                self.batch_idxs = np.random.choice(len(data_set.index),
                                                   len(data_set.index), replace=False)
                idxs = self.batch_idxs[self.batch_idx:self.batch_idx+batch_size]
                self.batch_idx += batch_size
            else:
                idxs = self.batch_idxs[self.batch_idx:self.batch_idx+batch_size]
                self.batch_idx += batch_size
                if self.batch_idx >= len(self.batch_idxs):
                    self.batch_idx = 0
                    print("Train All Users")

        else:
            idxs = np.random.choice(len(data_set.index),
                                    batch_size, replace=False)

        for i in idxs:
            data = data_set.iloc[i]
            user_id = torch.tensor(data.user_id)
            product_ids = data.product_id
            ratings = data.rating

            # subsamples
            support_ratings, support_product_ids, query_ratings, query_product_ids, normalized_num_samples = self.preprocess_wt_subsampling(
                product_ids, ratings)

            # make support set and query set
            support_data, rating_info = self.make_support_set(
                user_id, support_product_ids, support_ratings, normalized, use_label)

            query_data = self.make_query_set(
                user_id, query_product_ids, query_ratings)

            # make task information
            task_info = torch.Tensor(
                rating_info + self.task_info_num_samples*[normalized_num_samples])

            tasks.append((support_data, query_data, task_info))

        return tasks

    def make_pretraining_dataloader(self, df, batch_size=128, num_queries=1):
        '''
        funtion that makes dataloader for pretraining(single bert model)
        Args:
            df: data(train or valid)
            batch_size: training batch size
        return:
            dataloader: torch dataloader
        '''
        dataset = SequenceDataset(
            df, self.max_sequence_length, self.min_sub_window_size, self.default_rating, num_queries)
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
        self, df, max_len, min_sub_window_size=2, default_rating=0, num_queries=1
    ):
        """
        Args:
            df: preprocessed data
            max_len: max sequence length
            min_sub_window_size : minimum window size during subsampling
            default_rating: rating of padding
        """
        self.ratings_frame = df
        self.max_len = max_len
        self.min_sub_window_size = min_sub_window_size
        self.default_rating = default_rating
        self.num_queries = num_queries

    def __len__(self):
        return len(self.ratings_frame)

    def preprocessing(self, product_ids, ratings):
        """
            function that makes single random sequence for each user
            sequence length ranges from min_sub_window_size to max_len
        """
        seq_len = len(product_ids)
        maximum_len = seq_len if seq_len < self.max_len else self.max_len

        start_idx = np.random.randint(0, seq_len - maximum_len + 1)
        product_lst = []
        rating_lst = []
        for _ in range(self.num_queries):
            window_size = np.random.randint(
                self.min_sub_window_size, maximum_len+1)
            start_idx_im = np.random.randint(
                start_idx, maximum_len - window_size + start_idx + 1)
            product_ids_im = [0] * (self.max_len - window_size) + \
                list(product_ids[start_idx_im: start_idx_im+window_size])
            ratings_im = [0] * (self.max_len - window_size) + \
                list(ratings[start_idx_im: start_idx_im+window_size])
            product_ids_im = torch.LongTensor(product_ids_im).view(1, -1)
            ratings_im = torch.FloatTensor(ratings_im).view(1, -1)
            product_lst.append(product_ids_im)
            rating_lst.append(ratings_im)
        product_ids_f = torch.cat(product_lst, axis=0)
        ratings_f = torch.cat(rating_lst, axis=0)
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

        b, l = ratings.shape
        product_history = product_ids[:, :-1]
        target_product_id = product_ids[:, -1:]
        product_history_ratings = ratings[:, :-1]
        target_product_rating = ratings[:, -1:]
        user_id = user_id.repeat(
            self.num_queries, 1)

        return (user_id, product_history, target_product_id,  product_history_ratings), target_product_rating


# dataloader = DataLoader(args, pretraining=False)
# tasks = dataloader.generate_task(mode="train", batch_size=25, normalized=True)
# support, query, task = tasks[0]
# support_user_id, support_product_history, support_target_product, support_rating_history, support_target_rating = support
# query_user_id, query_product_history, query_target_product, query_rating_history, query_target_rating = query
# print(query_product_history)
# print(support_target_rating)
# print(query_rating_history)
# print(query_target_rating)
