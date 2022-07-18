from .meta_sasrec_model import MetaSAS4Rec
from .meta_narm_model import MetaNARM
from .meta_bert_model import MetaBERT4Rec
from .meta_grurec_model import MetaGRU4REC


MODELS = {
    'bert4rec': MetaBERT4Rec,
    'sas4rec': MetaSAS4Rec,
    'narm': MetaNARM,
    'gru4rec': MetaGRU4REC
}


def model_factory(args):
    model = MODELS[args.model]
    return model(args)
