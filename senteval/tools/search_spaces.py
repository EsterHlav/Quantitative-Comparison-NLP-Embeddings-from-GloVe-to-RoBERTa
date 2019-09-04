from hyperopt import hp
import hyperopt_changes
import hpb

SEARCH_SST2 = {
    'type':         hp.choice('type', [
        {
            'type':         'LogisticRegression'
        },
        {
            'type':         'MLP',
            'nb_layers':    hp.choice('nb_layers',  [1,2,3]),
            'act_fn':       hp.choice('act_fn', ['sigmoid', 'tanh', 'elu']),
            'nb_hid':       hp.quniform('nb_hid', 20, 100, 20),
            'dropout':      hp.uniform('dropout', 0.1, 0.6),
        }
    ]),
    'lr':           hpb.logbaseuniform('lr', -3, 0, base=10),
    'l2reg':        hpb.logbaseuniform('l2reg', -3, 0, base=10),
    'max_epoch':    hp.quniform('max_epoch', 20, 100, 10),

}

SEARCH_SST5 = SEARCH_SST2

SEARCH_SICKR = {
        'type':         hp.choice('type', [
            {
                'type':         'LogisticRegression'
            },
            {
                'type':         'MLP',
                'nb_layers':    hp.choice('nb_layers',  [1,2,3]),
                'act_fn':       hp.choice('act_fn', ['sigmoid', 'tanh', 'relu', 'elu']),
                'nb_hid':       hp.quniform('nb_hid', 10, 50, 10),
                'dropout':      hp.uniform('dropout', 0.1, 0.6),
            }
        ]),
        'lr':           hpb.logbaseuniform('lr', -3, 0, base=10),
        'l2reg':        hpb.logbaseuniform('l2reg', -3, 0, base=10),
        'max_epoch':    hp.quniform('max_epoch', 20, 100, 10),
    }

SEARCH_SICKE = {
        'type':         hp.choice('type', [
            {
                'type':         'LogisticRegression'
            },
            {
                'type':         'MLP',
                'nb_layers':    hp.choice('nb_layers',  [1,2,3]),
                'act_fn':       hp.choice('act_fn', ['sigmoid', 'tanh', 'elu']),
                'nb_hid':       hp.quniform('nb_hid', 10, 50, 10),
                'dropout':      hp.uniform('dropout', 0.1, 0.6),
            }
        ]),
        'lr':           hpb.logbaseuniform('lr', -3, 0, base=10),
        'l2reg':        hpb.logbaseuniform('l2reg', -3, 0, base=10),
        'max_epoch':    hp.quniform('max_epoch', 20, 100, 10),
        #'batch_size':   hpb.qlogbaseuniform('batch_size', 6, 10, q=1, base=2),
        #'optimizer':    hp.choice('optimizer', ['adam', 'rmsprop'])
    }

SEARCH_MRPC = {
        'type':         hp.choice('type', [
            {
                'type':         'LogisticRegression'
            },
            {
                'type':         'MLP',
                'nb_layers':    hp.choice('nb_layers',  [1,2]),
                'act_fn':       hp.choice('act_fn', ['sigmoid', 'tanh', 'elu']),
                'nb_hid':       hp.quniform('nb_hid', 20, 150, 20),
                'dropout':      hp.uniform('dropout', 0.1, 0.6),
            }
        ]),
        'lr':           hpb.logbaseuniform('lr', -3, 0, base=10),
        'l2reg':        hpb.logbaseuniform('l2reg', -3, 0, base=10),
        'max_epoch':    hp.quniform('max_epoch', 20, 100, 10),
    }

SEARCH_TREC = SEARCH_MRPC
SEARCH_TREC = SEARCH_MRPC
SEARCH_MPQA = SEARCH_MRPC
SEARCH_SUBJ = SEARCH_MRPC
SEARCH_MR = SEARCH_MRPC
SEARCH_CR = SEARCH_MRPC
