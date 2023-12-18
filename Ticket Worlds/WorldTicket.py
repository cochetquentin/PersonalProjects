from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import xml.etree.ElementTree as ET
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def xml_to_dataframe(xml_text):
    try:
        root = ET.fromstring(xml_text)
        
        data = []
        for element in root:
            row = {}
            for child in element:
                row[child.tag] = child.text
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        print(f"Erreur lors de la conversion XML en DataFrame : {str(e)}")
        return None
    
    
def formatName(name):
    name = name.split(" - ")[-1].split(" ~ ")[0]
    name = name.replace("스위스 스테이지", "Swiss Stage")
    name = name.replace("플레이-인 스테이지", "Play-in")
    return name


def getPlaceCode(GoodsCode):
    res = get(f"https://ticket.globalinterpark.com/Global/Play/CBT/GoodsInfo_CBT.asp?GoodsCode={GoodsCode}&lang=en&gatetype=CBT&device=pc&memNo=")

    for line in res.text.split("\n"):
        if line.__contains__('SetData("PlaceCode"'):
            return line.split('"')[-2]
        

def getPlaceInfos(GoodsCode, PlaceCode):
    res = get(f"https://ticket.globalinterpark.com/Global/Play/Goods/GoodsInfoXml.asp?Flag=RemainSeat&GoodsCode={GoodsCode}&PlaceCode={PlaceCode}&PlaySeq=001&LanguageType=G2001")
    return xml_to_dataframe(res.text)


def getSiteInfos():
    url = "https://www.globalinterpark.com/search-list?q=2023+League+of+Legends+World+Championship+&lang=ko"
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    dataSite = {"name":[], "GoodsCode":[], "PlaceCode":[], "link":[]}

    events = soup.findAll("li")

    for e in events:
        if e.text.startswith("2023 리그 오브 레전드 월드 챔피언십"):
            dataSite["name"].append(formatName(e.text))
            dataSite["GoodsCode"].append(e.a['href'].split("/")[-1])
            dataSite["PlaceCode"].append(getPlaceCode(dataSite["GoodsCode"][-1]))
            dataSite["link"].append(f"https://www.globalinterpark.com{e.a['href']}")

    dataSite = pd.DataFrame.from_dict(dataSite)
    dataSite.to_json("dataSite.json")


def getTicketInfos():
    try:
       dataSite = pd.read_json("dataSite.json")
    except:
        getSiteInfos()
        dataSite = pd.read_json("dataSite.json")

    dataSite["SeatInfos"] = dataSite.apply(lambda x: getPlaceInfos(x["GoodsCode"], x["PlaceCode"]), axis=1)
    dataSite["isFull"] = dataSite.apply(lambda x: x["SeatInfos"]["RemainCnt"].astype(int).sum() == 0, axis=1)

    return dataSite


def getAvailableSeats(df):
    seatInfos = df["SeatInfos"]

    infos = seatInfos[seatInfos["RemainCnt"].astype(int) >= 2][["SeatGradeName", "RemainCnt", "SalesPrice"]]
    
    if len(infos) > 0:
        return df["name"], infos, df["link"]
    else:
        return None
    

def getInteresedInfos(dataSite: pd.DataFrame):
    res = []
    filterData = dataSite[(dataSite["name"].str.contains("11.3")) & (~dataSite["isFull"])]
    for row in filterData.iterrows():
        info = getAvailableSeats(row[1])
        if info is not None:
            places = info[1][ (info[1]["SeatGradeName"] == "TIER5") & (info[1]["RemainCnt"].astype(int) > 1)]
            if len(places) > 0:
                res.append((info[0], places, info[2]))
    return res
    

def getMessage(infosList):
    resultat = ""
    for titre, dataframe, link in infosList:
        resultat += f"{titre}:\n"
        resultat += dataframe.to_string(index=False) + "\n"
        resultat += f"Lien: {link}\n\n"
    return resultat


def sendMail(corps_du_message):
    # Vos informations d'identification Gmail
    adresse_email = "YOUR_EMAIL"
    mot_de_passe = "YOUR_PASSWORD"
    destinataire = "YOUR_EMAIL"

    # Créez un objet MIMEMultipart pour le message
    message = MIMEMultipart()
    message["From"] = adresse_email
    message["To"] = destinataire  # Adresse e-mail du destinataire
    message["Subject"] = "Place pour les worlds disponibles"

    
    message.attach(MIMEText(corps_du_message, "plain"))

    # Établissez une connexion au serveur SMTP de Gmail
    serveur_smtp = smtplib.SMTP("smtp.gmail.com", 587)
    serveur_smtp.starttls()

    # Connectez-vous à votre compte Gmail
    serveur_smtp.login(adresse_email, mot_de_passe)

    # Envoyez l'e-mail
    texte_complet = message.as_string()
    serveur_smtp.sendmail(adresse_email, destinataire, texte_complet)

    # Déconnectez-vous du serveur SMTP
    serveur_smtp.quit()


def sendMailIfAvailableTickets():
    dataSite = getTicketInfos()
    infos = getInteresedInfos(dataSite)
    if len(infos) > 0:
        sendMail(getMessage(infos))
        return True
    else:
        return False
    

if __name__ == "__main__":
    while True:
        if sendMailIfAvailableTickets():
            print("Mail sent")
        else:
            print("No tickets available")
        time.sleep(30)