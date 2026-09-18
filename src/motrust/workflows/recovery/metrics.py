"""Feature-wise molecular recovery metrics with explicit tie handling."""
import numpy as np
from scipy.stats import rankdata

def macro_correlation(truth,pred,rank=False):
    """Same feature-wise statistic, bounded temporary memory for large cohorts."""
    values=[]
    for start in range(0,truth.shape[1],128):
        x=truth[:,start:start+128];y=pred[:,start:start+128]
        if rank:x=rankdata(x,axis=0);y=rankdata(y,axis=0)
        x=x.astype(np.float64);y=y.astype(np.float64);x-=x.mean(0);y-=y.mean(0)
        denominator=np.sqrt((x*x).sum(0)*(y*y).sum(0));valid=denominator>1e-12
        values.extend(((x*y).sum(0)[valid]/denominator[valid]).tolist())
    return (float(np.mean(values)) if values else np.nan),len(values)


def average_precision_columns(truth,pred):
    output=[]
    for start in range(0,truth.shape[1],256):
        y=truth[:,start:start+256];p=pred[:,start:start+256]
        positives=y.sum(0);valid=(positives>0)&(positives<len(y))
        if not valid.any():continue
        y=y[:,valid];p=p[:,valid];order=np.argsort(-p,axis=0,kind='stable')
        scores=np.take_along_axis(p,order,axis=0);ys=np.take_along_axis(y,order,axis=0)
        tp=np.cumsum(ys,axis=0,dtype=np.float64)
        ends=np.r_[scores[:-1]!=scores[1:],np.ones((1,scores.shape[1]),dtype=bool)]
        prior=np.r_[np.zeros((1,tp.shape[1])),np.maximum.accumulate(np.where(ends,tp,0),axis=0)[:-1]]
        ap=(np.where(ends,(tp-prior)*tp/np.arange(1,len(y)+1)[:,None],0).sum(0)/positives[valid])
        output.extend(ap.tolist())
    return float(np.mean(output)) if output else np.nan,len(output)


def metrics(truth,pred,modality):
    assert truth.shape==pred.shape and np.isfinite(pred).all()
    error=np.mean((truth-pred)**2,axis=1)
    pearson,n=macro_correlation(truth,pred);spearman,_=macro_correlation(truth,pred,rank=True)
    result={'rmse':float(np.sqrt(error.mean())),'mae':float(np.abs(truth-pred).mean()),'feature_pearson':pearson,'feature_spearman':spearman,'correlation_valid_features':n}
    if modality=='atac':
        ap,n=average_precision_columns(truth,pred);result.update(feature_auprc=ap,auprc_valid_features=n,positive_prevalence=float(truth.mean()),brier_score=float(error.mean()))
        counts=np.zeros(10,dtype=np.int64);psum=np.zeros(10);ysum=np.zeros(10)
        for start in range(0,truth.shape[1],128):
            p=pred[:,start:start+128];y=truth[:,start:start+128]
            bins=np.minimum((p*10).astype(np.int8),9)
            for b in range(10):
                mask=bins==b
                counts[b]+=mask.sum();psum[b]+=p[mask].sum(dtype=np.float64);ysum[b]+=y[mask].sum(dtype=np.float64)
        result['elementwise_ece_10bins']=float(np.abs(psum-ysum).sum()/truth.size)
    return result,error
