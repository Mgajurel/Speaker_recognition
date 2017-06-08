class PCAExtractor(object):
    def __init__(self, fs, win_length, win_shift_ms, n_pca,
    pre_emphasis_coefs):
        self.PRE_EMPH = pre_emphasis_coefs
        self.n_pca = n_pca
        #self.n_pca = n_pca + 1
        
        self.FRAME_LEN = int(float(win_length_ms) / 1000 * fs)
        self.FRAME_SHIFT = int(float(win_shift_ms) / 1000 * fs)
        self.window = hamming(self.FRAME_LEN)
        
    # function to convert pca to cepstrum coefficient
    def pca_to_cc(self, pca):
        pass
        
    def pcac(self, signal):
        pass
        
        
    #Function to extract pca from given signal    
    def extract(self, signal):
        pass
        
        
