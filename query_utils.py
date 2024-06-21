
import torch


from transformers import AutoTokenizer, AutoModel



def build_model(cfg):
    """
    """
    cfg.logger.debug("Loading model...")
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckpt)
    model = AutoModel.from_pretrained(cfg.model_ckpt)
    model.to(device)

    def cls_pooling(model_output):
        """
        Use the 'cls' vector from the models output as the encoding vector
        """
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(text_list):
        encoded_input = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = model(**encoded_input)
        return cls_pooling(model_output)


    cfg.tokenizer = tokenizer
    cfg.cls_pooling = cls_pooling
    cfg.get_embeddings = get_embeddings
    return model


