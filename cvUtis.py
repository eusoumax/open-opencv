

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
        #verificar se na string tem uma url
        if isPadraoHttp(img):            
            img = urlParaImagem(img)                       
        else:
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
    if isinstance(img,str): 
        #verificar se na string tem uma url
        if isPadraoHttp(img):            
            img = urlParaImagem(img)
        else:
            img = cv2.imread(img)
    
    if img is None:
        print('img é nula')  
        return None
    return  cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    
    
def imagemParaRGB(img):
    import cv2
    if isinstance(img,str): 
        #verificar se na string tem uma url
        if isPadraoHttp(img):            
            img = urlParaImagem(img)
        else:
            img = cv2.imread(img)
    
    if img is None:
        print('img é nula')  
        return None
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def mostraImagemComReconhecimentoFacialCascade(img,escala=(20,10)):
    import cv2
    if isinstance(img,str): 
        #verificar se na string tem uma url
        if isPadraoHttp(img):            
            img = urlParaImagem(img)                       
        else:
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

def listarRostosImagemCascade(img,escala=(20,10)):
    import cv2
    if isinstance(img,str): 
        #verificar se na string tem uma url
        if isPadraoHttp(img):            
            img = urlParaImagem(img)                       
        else:
            img = cv2.imread(img)
    
    if img is None:
        print('img é nula')  
    
    else:
        imgCinza = imagemParaCinza(img)
        facesEncontradas = buscarFacesCascade(img)
        listaRostos = []
        if len(facesEncontradas) >0:
            
            for face in facesEncontradas:
                listaRostos.append(imagemRostoCascade(img,face))                       
                
        return listaRostos
    
    return None
            
def salvarImagem(caminho,nomeArquivo,imagemCV):
    import os,cv2
    arquivoNovo = os.path.join(caminho,nomeArquivo)
    cv2.imwrite(arquivoNovo,imagemCV)
    
def imagemRostoCascade(img,faceEncontrada):
    x,y,l,a = faceEncontrada    
    return img[y:y+a,x:x+l]


def urlParaImagem(url):
    #https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
    import numpy as np
    import urllib.request
    import cv2
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    with urllib.request.urlopen(url, timeout = 5) as site:
        resp = site.read()
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image    
    return image

def isPadraoHttp(url):
    import re
    pattern = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    
    regex = re.compile(pattern)
    match = regex.match(url)
    if match:
        #print('isPadraoHttp')
        return True
    else:
        return False
    
def salvarImagemUrl(url,caminho,nomeArquivo):
    import os
    import urllib.request
    arquivo = os.path.join(caminho,nomeArquivo)
    try:
        urllib.request.urlretrieve(url,arquivo)
    except Exception as ex:
        print('erro no salvarImagemUrl',ex)
        
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

def linhasColunas(registros,colunas=5):
    from math import floor
    resto  = 0
    linhas = 0
    resto  =registros % colunas   
    divisao = floor(registros  / colunas)
    if resto > 0:
        linhas = divisao + 1
    else:
        linhas = divisao 
    #print(linhas, colunas)
    return linhas, colunas

def mostrarRostosImagemCascade(img,colunas=3):   
    lista = listarRostosImagemCascade(img)
    
    registros = len(lista)
    
    
    if registros == 0:
        print('sem rostos encontrados')
        return
    
    np.random.seed(19680801)
    #Nr = 3
    #Nc = 2
    Nr, Nc = linhasColunas(registros,colunas)
    cmap = "cool"
    plt.figure(figsize=(20,10))

    if registros == 1:
        fig, axs = plt.subplots(1,figsize=(5,3))
        #print(axs)

    elif(registros <= Nc):
        fig, axs = plt.subplots(Nc,figsize=(5,4))
        #print(axs)
    else:
        fig, axs = plt.subplots(Nr, Nc,figsize=(15,15))
    fig.suptitle('Multiplas Faces')



    images = []
    count = 0

    if registros == 1:
        images.append(axs.imshow(imagemParaRGB(lista[count]), cmap=cmap))
        axs.label_outer()
        count+=1

    elif Nc == 1:       
        for i in range(registros):
            if count<registros :
                images.append(axs[i].imshow(imagemParaRGB(lista[count]), cmap=cmap))
                axs[i].label_outer()
                count+=1

    elif registros <= Nc:       
        for i in range(Nc):
            if count<registros :
                images.append(axs[i].imshow(imagemParaRGB(lista[count]), cmap=cmap))
                axs[i].label_outer()
                count+=1
    else:
        for i in range(Nr):
            for j in range(Nc):
                if count<registros :
                    images.append(axs[i, j].imshow(imagemParaRGB(lista[count]), cmap=cmap))
                    axs[i, j].label_outer()
                    count+=1



    plt.show()        
def imagemTelaDesktop():
    #https://www.pyimagesearch.com/2018/01/01/taking-screenshots-with-opencv-and-python/
    import pyautogui,cv2
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def salvarImagemTelaDesktop(caminho,nomeArquivo):    
    imagemDesktop = imagemTelaDesktop()
    salvarImagem(caminho,nomeArquivo,imagemDesktop)

def mostraImagemTelaDesktop():
    mostraImagem(imagemTelaDesktop())
    
    
def mostrarContornosImagem(img):
    import cv2
    import numpy as np
    import imutils  
    img = imagemParaRGB(img)
    original = img.copy()

    #outro teste
    gray = imagemParaCinza(img)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    font = cv2.FONT_HERSHEY_COMPLEX


    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        cv2.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 3:            
            cv2.putText(img, "Triangle", (x, y), font, 1, (0))
            
        elif len(approx) == 4:            
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
        elif len(approx) == 5:            
            cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
            pass
        elif 6 < len(approx) < 15:
            cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
            
        else:
            cv2.putText(img, "Circle", (x, y), font, 1, (0))            
    mostraImagem(img)
    
def listaUrlImagens():
    umchemba01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/6579744e8293106ff7d3f74650a2fc3e/5D58F81E/t51.2885-15/sh0.08/e35/p640x640/50854114_382666809195801_5792591852174946727_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'

    umchemba02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/8e0bf7fe6046cef817c35f4204ada1ae/5D6776D4/t51.2885-15/e35/41705515_327926961298610_8338125433094652131_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    umchemba03 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/9445aa7571775b1e729804902d6ff7d9/5D9C1C0D/t51.2885-15/sh0.08/e35/p750x750/41517301_402299396971242_2441153836235470818_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    umchemba04 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/347cb07719ae5652e9c9158747c1c59e/5D9E6490/t51.2885-15/sh0.08/e35/s750x750/19764348_461890640842970_9219768318775263232_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    umchemba05 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/56ce7af1743f912b97a4972d65e21beb/5D78A512/t51.2885-15/e35/24331649_2061345580766876_2693149441841430528_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    umchemba06 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/693e267dfd6eeb767f38fecaef0e439a/5D67B746/t51.2885-15/e35/32161800_413673555773681_16210091338366976_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    amirahdyme01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/a9f774d2261e15b4bd5839f94b3904c1/5D5E342A/t51.2885-15/e35/47583829_135783594090549_412509093972710360_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    amirahdyme02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/e558e137b7a3871bbe4bf8c2153a6ac3/5D7B46A0/t51.2885-15/e35/46860669_346252006174186_7812623847618288731_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    iamsodeelishis01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/e558e137b7a3871bbe4bf8c2153a6ac3/5D7B46A0/t51.2885-15/e35/46860669_346252006174186_7812623847618288731_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    iamsodeelishis02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/fa0bf27580d6f6c22527fc65e079a298/5D7CD35D/t51.2885-15/sh0.08/e35/s750x750/42471811_316724165790702_4478489634736751374_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    iamsodeelishis03 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/f93d248f7683f48d56f04855be955929/5D9D44DF/t51.2885-15/sh0.08/e35/p750x750/41620665_1058832567610462_2756353141187280896_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    melovingme___01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/9ac302b88ce8240962ee85479e83486f/5D68C903/t51.2885-15/e35/33043043_650284665312013_7503130608516726784_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    melovingme___02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/fb6198094d49a0ff1825bdf21869a568/5D7A34B6/t51.2885-15/sh0.08/e35/p750x750/35339813_188696125147475_4018523375260401664_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    the_nikkinicole01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/1a567ca96b925ebf9c34ca714a520fc2/5D65965C/t51.2885-15/sh0.08/e35/p750x750/41602425_1857425967637950_2371456061010714654_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    the_nikkinicole02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/ab11c0cbf0101c0e41dc165ed164b8ad/5D659735/t51.2885-15/e35/40479858_1870585759692468_4656558854332550454_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    leealonso01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/b371fa87e6dc347afbde4e980a06e2dd/5D5F2648/t51.2885-15/sh0.08/e35/p640x640/54800812_131997741270443_6006828178096260471_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    leealonso02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/cd04b86a2416f34ed32bf4a1d14f1518/5D7DA4F4/t51.2885-15/sh0.08/e35/s640x640/36834330_1936094193117568_7791079496806825984_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    leealonso03 =  'https://instagram.fqnv1-1.fna.fbcdn.net/vp/41c596dbefd227984117a96686768207/5D6CBE4C/t51.2885-15/e35/33559606_286586241882634_1438682618160742400_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    newyorktaylor01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/fba7e3fb0314468e9bbd95c6b9d20ca6/5D64E24C/t51.2885-15/e35/56323837_410871736367355_4188698656850875555_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    newyorktaylor02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/d453b68456d9d1b7184ebea8d916fd42/5D6F0A5C/t51.2885-15/e35/51434927_1257428531079587_6293313108439778579_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    newyorktaylor03= 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/1a334402d9828a45d1e1f3dd89635688/5D6DFE0C/t51.2885-15/e35/44667298_439125846617939_2728318745670429431_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    stefaniejoosten01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/2fad8732624f9d6b1faf48448cea0c0e/5D7001D7/t51.2885-15/e35/46202600_508669809625100_8726757882133459396_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    stefaniejoosten02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/3c299e7d6460fa1ac3aaf0203c4a15a2/5D5F6CF5/t51.2885-15/e35/28750743_1606343122793058_3086041281869119488_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    camilabrandaooo01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/e2880b1df92c832335bcc38e2e1455e3/5D63F687/t51.2885-15/e35/43235350_506412633206286_8373262784287631614_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    camilabrandaooo02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/e2880b1df92c832335bcc38e2e1455e3/5D63F687/t51.2885-15/e35/43235350_506412633206286_8373262784287631614_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    _stelle_01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/136085a3fd3f4a15c58bdd97ba30ad51/5D5F2841/t51.2885-15/e35/43121435_244084982905634_7718917782760646279_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    leticialongati01 ='https://instagram.fqnv1-1.fna.fbcdn.net/vp/1199d6a7f2c56d343fbb5c4a6bb5f378/5D67C81F/t51.2885-15/e35/56344621_160375391625704_8421152222076486131_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    leticialongati02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/fd8c7ff0c26481e95cf8a912a3408b2a/5D601B12/t51.2885-15/e35/53077579_748728855511502_5497072464804482663_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    leticialongati03 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/e9b03124fb3197395376471007e45894/5D65E5B7/t51.2885-15/e35/50797113_289105348421417_1140430543855649274_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    deniserocha01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/70c0ad0dab6dc2f5327fa2ae7fa908c7/5D7C4B6C/t51.2885-15/e35/22710893_1464547063661106_6391195332570513408_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    deniserocha02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/cf6175106210ce254610136ecd13d5df/5D9E44C2/t51.2885-15/e35/15625227_1512112948806711_8987543362680651776_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    bby_cai01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/f84d22c9e44b02b155a9eb61cb1f3ea3/5D70A862/t51.2885-15/e35/49858447_1286867541437988_1533474748019843811_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    bby_cai02 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/117fa72c4267b89530ae3571a59e7a44/5D5E07BB/t51.2885-15/e35/44291418_1608553469452200_108482183649848813_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    bby_cai03 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/daa2fda07beef0e76f0e35ae23dcb8f6/5D6B813D/t51.2885-15/sh0.08/e35/p640x640/46789634_209409879941245_5967224204283180637_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    a1_fleeky01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/80164ae251d242d0b3f8e30a46e94e1e/5D58B04A/t51.2885-15/e35/40197955_311476146071511_6494304986973194587_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    yeimmyoficial01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/1ccc2555f58fbbc15193b44d66fd85e2/5D58F945/t51.2885-15/e35/40075594_280042316151322_481523609781665792_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    mayaradellavega01 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/de63be68279548144fea40bcc08825a2/5D6D0721/t51.2885-15/e35/38729208_286716968772804_8776248250370883584_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    marthamurillo1901 = 'https://instagram.fqnv1-1.fna.fbcdn.net/vp/642bb127bea31029a9f76856cec9dbc4/5D57EE73/t51.2885-15/sh0.08/e35/p750x750/37955034_2104433649830559_4199845551488892928_n.jpg?_nc_ht=instagram.fqnv1-1.fna.fbcdn.net'
    lista =[
        umchemba01,
        umchemba02,
        umchemba03,
        umchemba04,
        umchemba05,
        umchemba06,
        amirahdyme01,
        amirahdyme02,
        iamsodeelishis01,
        iamsodeelishis02,
        iamsodeelishis03,
        melovingme___01,
        melovingme___02,
        the_nikkinicole01,
        the_nikkinicole02,
        leealonso01,
        leealonso02,
        leealonso03,
        newyorktaylor01,
        newyorktaylor02,
        newyorktaylor03,
        stefaniejoosten01,
        stefaniejoosten02,
        _stelle_01,
        camilabrandaooo01,
        camilabrandaooo02,
        leticialongati01,
        leticialongati02,
        leticialongati03,
        deniserocha01,
        deniserocha02,
        bby_cai01,
        bby_cai02,
        bby_cai03,
        a1_fleeky01,
        yeimmyoficial01,
        mayaradellavega01,
        marthamurillo1901        
    ]
    
###criacao de metodos para biblioteca dlib

def buscarFacesDlib(img):
    import dlib,cv2
    detectorFace = dlib.get_frontal_face_detector()    
    imgCinza = imagemParaCinza(img)
    facesEncontradas = detectorFace(imgCinza,2)    
    return facesEncontradas

def montaRetangulosDlib(imagem,facesEncontradas):    
    import cv2
    for face in facesEncontradas:
        e,t,d,b = recuperaPosicoesFaceDlib(face)
        cv2.rectangle(imagem,(e,t),(d,b),(192,0,0),2)
        
def imagemRostoDlib(img,facesEncontradas):
    e,t,d,b = facesEncontradas    
    return img[e:d,e:b]        
def recuperaPosicoesFaceDlib(face):
    return (int(face.left()),int(face.top()),int(face.right()),int(face.bottom()))

def imagemRostoDlib(img,faceEncontrada):
    x,y,l,a = recuperaPosicoesFaceDlib(faceEncontrada)    
    return img[y:a,x:l]

def mostraImagemComReconhecimentoFacialDlib(img,escala=(20,10)):
    import cv2
    if isinstance(img,str): 
        #verificar se na string tem uma url
        if isPadraoHttp(img):            
            img = urlParaImagem(img)                       
        else:
            img = cv2.imread(img)
    
    if img is None:
        print('img é nula')  
    
    else:        
        facesEncontradas = buscarFacesDlib(img)        
        if len(facesEncontradas) >0:
            montaRetangulosDlib(img,facesEncontradas)
            mostraImagem(img,escala)
            
        else:
            mostraImagem(img,escala)
            
def listarRostosImagemDlib(img,escala=(20,10)):
    import cv2
    if isinstance(img,str): 
        #verificar se na string tem uma url
        if isPadraoHttp(img):            
            img = urlParaImagem(img)                       
        else:
            img = cv2.imread(img)
    
    if img is None:
        print('img é nula')  
    
    else:
        imgCinza = imagemParaCinza(img)
        facesEncontradas = buscarFacesDlib(img)
        listaRostos = []
        if len(facesEncontradas) >0:
            
            for face in facesEncontradas:
                listaRostos.append(imagemRostoDlib(img,face))                       
                
        return listaRostos
    
    return None            
def mostrarRostosImagemDlib(img,colunas=3):   
    lista = listarRostosImagemDlib(img)
    
    registros = len(lista)    
    
    if registros == 0:
        print('sem rostos encontrados')
        return
    
    np.random.seed(19680801)
    #Nr = 3
    #Nc = 2
    Nr, Nc = linhasColunas(registros,colunas)
    cmap = "cool"
    plt.figure(figsize=(20,10))

    if registros == 1:
        fig, axs = plt.subplots(1,figsize=(5,3))
        #print(axs)

    elif(registros <= Nc):
        fig, axs = plt.subplots(Nc,figsize=(5,4))
        #print(axs)
    else:
        fig, axs = plt.subplots(Nr, Nc,figsize=(15,15))
    fig.suptitle('Multiplas Faces')



    images = []
    count = 0

    if registros == 1:
        images.append(axs.imshow(imagemParaRGB(lista[count]), cmap=cmap))
        axs.label_outer()
        count+=1

    elif Nc == 1:       
        for i in range(registros):
            if count<registros :
                images.append(axs[i].imshow(imagemParaRGB(lista[count]), cmap=cmap))
                axs[i].label_outer()
                count+=1

    elif registros <= Nc:       
        for i in range(Nc):
            if count<registros :
                images.append(axs[i].imshow(imagemParaRGB(lista[count]), cmap=cmap))
                axs[i].label_outer()
                count+=1
    else:
        for i in range(Nr):
            for j in range(Nc):
                if count<registros :
                    images.append(axs[i, j].imshow(imagemParaRGB(lista[count]), cmap=cmap))
                    axs[i, j].label_outer()
                    count+=1



    plt.show()        
    
    
def videoParaImagens(caminhoArquivo,nomeArquivo,destinoArquivos):
    import os
    arquivo = os.path.join(caminhoArquivo,nomeArquivo)
    video = cv2.VideoCapture(arquivo)
    success, image = video.read()
    count = 0
    success = True
    while success:
        success, image = video.read()
        fNomeArquivo = "imagem{:04d}".format(count)
        novoArquivo = os.path.join(destinoArquivos,fNomeArquivo)
        cv2.imwrite(novoArquivo, image)  # save frame as JPEG file
        print('Read a new frame: ', success)
        count += 1
        
def copiarRostosDeDiretorioImagens(diretorioOrigem,diretorioDestino):
    import os,glob, cv2
    
    dirOrigem = os.path.join(diretorioOrigem)
    isDiretorioOrig = os.path.isdir(dirOrigem)
    isDiretorioDest = os.path.isdir(diretorioDestino)
    if isDiretorioOrig == False:
        print('diretorio origem invalido')
        return
        
    if isDiretorioDest == False:
        print('diretorio destino invalido')    
        return
    
    
    listaArquivosOrigem = [f for f in glob.glob(os.path.join(diretorioOrigem,'*.jpg'))]
    count = 0
    for arquivo in listaArquivosOrigem:
        listaRostos = utils.listarRostosImagemDlib(arquivo)
        
        if listaRostos is not None and len(listaRostos) > 0:
            for rosto in listaRostos:
                vNome = "rosto{:04d}.jpg".format(count)
                nomeArquivo = os.path.join(diretorioDestino,vNome)
                cv2.imwrite(nomeArquivo,rosto)
                count+=1
                print('salvando {}'.format(nomeArquivo))
        
    #print(listaArquivosOrigem,dirOrigem)
    