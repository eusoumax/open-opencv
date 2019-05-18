

def listarArquivos(diretorioEPadraoPesquisaArquivos):
    import glob
    lista =  [f for f in glob.glob(diretorioEPadraoPesquisaArquivos)]
    
    
def buscarFacesCascade(img):
    import cv2
    reconhecimento_facial_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces_encontradas = reconhecimento_facial_cascade.detectMultiScale(img,
                                                                   scaleFactor=1.1,
                                                                   minNeighbors=5)
    return faces_encontradas
    


def mostraImagem(img,escala=(20,10)):    
    import matplotlib.pyplot as plt    
    import cv2
    if isinstance(img,str):
        img = cv2.imread(img)
    
    if img is None:
        print('img é nula')  
    else:
        plt.figure(figsize=escala)
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  
        
        
def monta_retangulos_cascade(imagem,facesEncontradas):    
    import cv2
    for e,t,d,b in facesEncontradas:
        cv2.rectangle(imagem,(e,t),(e+d,t+b),(192,0,0),2)
        
def imagemParaCinza(img):
    import cv2
    return  cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    
    
def imagemParaRGB(img):
    import cv2
    return cv2.cvtColor(imagem02,cv2.COLOR_BGR2RGB)

def mostraImagemComReconhecimentoFacialCascade(img,escala=(20,10)):
    import cv2
    if isinstance(img,str):
        img = cv2.imread(img)
    
    if img is None:
        print('img é nula')  
    
    else:
        imgCinza = imagemParaCinza(img)
        facesEncontradas = buscarFacesCascade(img)
        
        if len(facesEncontradas) >0:
            monta_retangulos_cascade(img,facesEncontradas)
            mostraImagem(img,escala)
            
        else:
            mostraImagem(img,escala)
        