from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Lidstone

def train_and_test_perplexity(train_corpus, test_corpus, n_order=2):
    train_data, padded_vocab = padded_everygram_pipeline(n_order, train_corpus)
    
    model = Lidstone(order=n_order, gamma=0.1)
    model.fit(train_data, padded_vocab)
    
    test_data, _ = padded_everygram_pipeline(n_order, test_corpus)
    
    total_ppl = 0
    count = 0
    for sent in test_data:
        try:
            ppl = model.perplexity(sent)
            if ppl < float('inf'):
                total_ppl += ppl
                count += 1
        except:
            continue
            
    return total_ppl / count if count > 0 else float('inf')