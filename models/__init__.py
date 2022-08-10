from .meta_sasrec_model import MetaSASRec
from .meta_narm_model import MetaNARM
from .meta_bert_model import MetaBERT4Rec
from .meta_grurec_model import MetaGRU4REC
from .meta_ncf_model import MetaNCF


MODELS = {
    'bert4rec': MetaBERT4Rec,
    'sasrec': MetaSASRec,
    'narm': MetaNARM,
    'gru4rec': MetaGRU4REC,
    'ncf': MetaNCF
}


def model_factory(args):
    model = MODELS[args.model]
    return model(args)
