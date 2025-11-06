import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from msalde.esm_util import get_esm_model_and_alphabet

AAorder=['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

##### INFERENCE
def load_esm_model(model_name,device='cpu'):
    model, alphabet = get_esm_model_and_alphabet(model_name)
    model = model.to(device) 
    repr_layer = int(model_name.split('_')[1][1:])
    batch_converter = alphabet.get_batch_converter() 
    return model.eval().to(device),alphabet,batch_converter,repr_layer

def get_wt_LLR(input_df,model,alphabet,batch_converter,device='cpu',silent=False): 
    # input: df.columns= id,	gene,	seq, length
    # make sure input_df does not contain any nonstandard amino acids
    AAorder=['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']
    genes = input_df.id.values
    LLRs=[];input_df_ids=[]
    for gname in tqdm(genes,disable=silent):
        seq_length=input_df[input_df.id==gname].length.values[0]
    
        dt = [(gname+'_WT',input_df[input_df.id==gname].seq.values[0])]
        batch_labels, batch_strs, batch_tokens = batch_converter(dt)
        with torch.no_grad():
            results_ = torch.log_softmax(model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)['logits'],dim=-1)

        WTlogits = pd.DataFrame(results_[0,:,:].cpu().numpy()[1:-1,:],columns=alphabet.all_toks,index=list(input_df[input_df.id==gname].seq.values[0])).T.iloc[4:24].loc[AAorder]
        WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
        wt_norm=np.diag(WTlogits.loc[[i.split(' ')[0] for i in WTlogits.columns]])
        LLR = WTlogits - wt_norm

        LLRs += [LLR]
        input_df_ids+=[gname]

    return input_df_ids,LLRs

def get_logits(seq,model,batch_converter,format=None,device=0):
    data = [ ("_", seq),]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],dim=-1).cpu().numpy()
    if format=='pandas':
        WTlogits = pd.DataFrame(logits[0][1:-1,:],columns=alphabet.all_toks,index=list(seq)).T.iloc[4:24].loc[AAorder]
        WTlogits.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(WTlogits.columns)]
        return WTlogits
    else:
        return logits[0][1:-1,:]

def get_PLL(seq,model,alphabet,batch_converter,reduce=np.sum,device='cpu'):
    s=get_logits(seq,model=model,batch_converter=batch_converter,device=device)
    idx=[alphabet.tok_to_idx[i] for i in seq]
    return reduce(np.diag(s[:,idx]))

### MELTed CSV
def meltLLR(LLR,savedir=None):
    vars = LLR.melt(ignore_index=False)
    vars['variant'] = [''.join(i.split(' '))+j for i,j in zip(vars['variable'],vars.index)]
    vars['score'] = vars['value']
    vars = vars.set_index('variant')
    vars['pos'] = [int(i[1:-1]) for i in vars.index]
    del vars['variable'],vars['value']
    if savedir is not None:
        vars.to_csv(savedir+'var_scores.csv')
    return vars


## stop gain variant score
def get_minLLR(seq,stop_pos):
    return min(get_wt_LLR(pd.DataFrame([('_','_',seq,len(seq))],columns=['id','gene','seq','length'] ),silent=True)[1][0].values[:,stop_pos:].reshape(-1))


# ############### EXAMLE ##################
# ## Load model
model,alphabet,batch_converter,repr_layer = load_esm_model(model_name='esm2_t6_8M_UR50D',device='cpu')
# ## Create a toy dataset
df_in = pd.DataFrame([('P1','gene1','FISHWISHFQRCHIPSTHATARECRISP',28),
                      ('P2','gene2','RAGEAGAINSTTHEMACHINE',21),
                      ('P3','gene3','SHIPSSAILASFISHSWIM',19),
                      ('P4','gene4','A'*1948,1948)], columns = ['id','gene','seq','length'])
# ## Get LLRs
ids,LLRs = get_wt_LLR(df_in, model, alphabet, batch_converter, device='cpu', silent=False)
for i,LLR in zip(ids,LLRs):
    print(i,LLR.shape)
# ## Get PLL
pll = get_PLL(df_in.seq.values[0], model, alphabet, batch_converter, device='cpu')
pause
# ## indel: 14_IPS_delins_EESE (FISHWISHFQRCHIPSTHATARECRISP --> FISHWISHFQRCHEESETHATARECRISP)
# get_PLLR('FISHWISHFQRCHIPSTHATARECRISP','FISHWISHFQRCHEESETHATARECRISP',14)
# ## stop at position 17
# get_minLLR(df_in.seq.values[0],17)