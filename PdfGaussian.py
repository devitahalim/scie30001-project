import numpy as np

def GaussianPDF(data, mean:float, var:float):
    pdf_gauss=(1/(np.sqrt(2*np.pi*var)))*np.exp(-(np.square(data - mean)/(2*var)))
    return pdf_gauss