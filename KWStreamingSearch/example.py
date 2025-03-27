# If you want to do inference for ctc-based or transducer-based acoustic models, you can refer to the following code:
import torch
import torchaudio

from KWStreamingSearch.CTC.ctc_streaming_search import CTCFsdStreamingSearch

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stage1: load model
# your_model = torch.load('your_model.pth') # Transducer or CTC model
# your_model.eval()
# your_model.to('cuda')
your_model = torch.nn.Linear(40, 71).to(device)

# Stage2: load wav and extract the features.
test_audio = torch.randn(1, 16000).to(device) # 10s of 16kHz audio
fbank = torchaudio.compliance.kaldi.fbank(
    test_audio,
    num_mel_bins=40,
    frame_length=25,
    frame_shift=10,
    sample_frequency=16000,
) # T, 40

batch_fbank = fbank.unsqueeze(0) # (B=1, T, 40)
fbank_length = torch.tensor([fbank.shape[1]]).to(device) # (B=1, ), for decoding, we only support B=1.

# Stage3: define the keyword.
# phoneme or subword sequence of the keyword
keyword_sequence = torch.tensor([[1, 2, 3, 4, 5]]).to(device) # (B=1, U), U is the length of the keyword sequence
keyword_sequence_length = torch.tensor([keyword_sequence.shape[1]]).to(device) # (B=1, )

# Stage4: model forward to get logits.
# CTC model: (B=1, T, V), V is the vocab size with blank.
# Transducer model: (B=1, T, U, V), V is the vocab size with blank (also including durations, if use TDT.).
your_model = torch.nn.Linear(40, 71).to(device)
logits = your_model(batch_fbank)

# Stage5: define the corresponding search method. We use the CTC fsd (frame-synchronous decoding) search as an example.
# !!!Caution!!!: KWS decoding contains a log-softmax calculation, so please pass logits to the search instead of posterior.
kws_streaming_search = CTCFsdStreamingSearch(blank=0) # some frameworks doesn't use blank=0, maybe blank=len(vocab). 
_, score_tlist, _, _ = kws_streaming_search(
    logits, keyword_sequence, fbank_length, keyword_sequence_length
) # score_tlist is a list of the keyword activation scores at each time step t.

# Stage 6: define the threshold to get the keyword activation status.
threshold = 0.9
for time_step, score in enumerate(score_tlist):
    if score > threshold:
        print(f"Keyword detected at time step {time_step}")