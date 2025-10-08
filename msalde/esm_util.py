import gc
import esm


def get_esm_model_and_alphabet(base_model_name: str):
    """
    We garbage collect previous model/tokenizer if any to free memory
    on small memory systems.
    We assume that caller has dropped references to previous model/tokenizer.
    """
    model_name = base_model_name.replace("facebook/","")
    gc.collect()
    # torch.cuda.empty_cache()
    model, alphabet = esm.pretrained.__dict__[model_name]()
    return model, alphabet


