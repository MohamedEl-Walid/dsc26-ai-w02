import numpy as np
import pandas as pd
import pytest

from rawaj.audit import audit


def run(df):
    return audit(df, target='label', id_col='id', time_col='time')


def test_counts_blanks_duplicates_and_no_mutation():
    df = pd.DataFrame({'id': ['b','a','a','a', ' \t'], 'time': ['2026-09-01 10:00','2026-02-30 10:00','','',None], 'label':[1,0,None,None,1], 'channel':['app',' ',None,None,'\n'], 'currency':['EGP',None,'EGP','EGP','EGP']})
    original = df.copy(deep=True)
    result = run(df)
    assert set(result) == {'n_rows','missing','duplicate_ids','duplicate_rows','constant_columns','unlabelled_rows','positive_rate','bad_timestamps','time_range','leakage_suspects'}
    assert result['n_rows'] == 5
    assert result['duplicate_ids'] == ['a']
    assert result['duplicate_rows'] == 1
    assert result['missing']['channel'] == .8
    assert result['unlabelled_rows'] == 2
    assert result['positive_rate'] == 2/3
    assert result['bad_timestamps'] == 1
    assert result['constant_columns'] == ['channel','currency']
    assert pd.Timestamp(result['time_range'][0]) == pd.Timestamp('2026-09-01 10:00', tz='UTC')
    pd.testing.assert_frame_equal(df, original)


def test_leakage_values_inverse_missingness_and_constant_difference():
    y = np.tile([0,1], 30)
    df = pd.DataFrame({'id':np.arange(60), 'time':['2026-09-01 10:00']*60, 'label':y, 'copy':y, 'inverse':1-y, 'closed':[None if v==0 else '2026-09-10' for v in y], 'honest':np.tile([0,0,1,1],15), 'constant':['EGP']*60, 'empty':[None]*60})
    r = run(df)
    assert r['leakage_suspects'] == ['closed','copy','inverse']
    assert 'closed' in r['constant_columns']
    assert 'empty' in r['constant_columns']


def test_missingness_leak_survives_one_class_present_values():
    df = pd.DataFrame({'id':range(40),'time':['2026-09-01']*40,'label':[0]*20+[1]*20, 'numeric':[np.nan]*20+list(range(20))})
    assert run(df)['leakage_suspects'] == ['numeric']


def test_small_sample_and_one_class_skip_leakage():
    for y in ([0]*30, [1]*30, [0,1]*9):
        df = pd.DataFrame({'id':range(len(y)), 'time':['bad']*len(y), 'label':y, 'copy':y})
        assert run(df)['leakage_suspects'] == []


def test_numeric_minimum_and_unlabelled_exclusion():
    # Only 19 value observations, with balanced missingness: no suspect.
    y = [0,1]*20
    values = list(range(19)) + [None]*21
    df = pd.DataFrame({'id':range(40),'time':['bad']*40,'label':y,'feature':values})
    assert run(df)['leakage_suspects'] == []
    extra = pd.DataFrame({'id':range(40,80),'time':['bad']*40,'label':[None]*40,'feature':[1]*40})
    assert run(pd.concat([df,extra],ignore_index=True))['leakage_suspects'] == []


def test_empty_unlabelled_and_missing_columns():
    empty = pd.DataFrame(columns=['id','time','label'])
    r = run(empty)
    assert r['n_rows'] == 0 and r['positive_rate'] == 0
    assert r['time_range'] == [None,None]
    assert r['missing'] == dict(id=0.0,time=0.0,label=0.0)
    df = pd.DataFrame({'id':[None,' '],'time':['','invalid'],'label':[None,'\t']})
    r = run(df)
    assert r['unlabelled_rows'] == 2 and r['positive_rate'] == 0
    assert r['bad_timestamps'] == 1 and r['duplicate_ids'] == []
    for col in ['id','time','label']:
        with pytest.raises(KeyError):
            run(df.drop(columns=col))


def test_nullable_numeric_boolean_columns():
    y = [0,1]*20
    df = pd.DataFrame({'id':range(40),'time':['2026-09-01']*40,'label':pd.Series(y,dtype='Int64'), 'bool_leak':pd.Series(y,dtype='boolean'), 'nullable_leak':pd.Series(y,dtype='Int64')})
    assert run(df)['leakage_suspects'] == ['bool_leak','nullable_leak']
