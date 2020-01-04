import os

class Landmark:
    def __init__(self, name, path):
        self.name = name
        self.path = path

def getLandmarks():
    landmarks = []
    root = "thumbnails"
    
    landmarks.append(Landmark("Anıtkabir", os.path.join(root, "anıtkabir.jpg")))
    landmarks.append(Landmark("Arc de Triomphe", os.path.join(root, "arc_de_triomphe.jpg")))
    landmarks.append(Landmark("Hagia Sophia", os.path.join(root, "ayasofya.jpg")))
    landmarks.append(Landmark("Beylerbeyi Palace", os.path.join(root, "beylerbeyi_sarayi.jpg")))
    landmarks.append(Landmark("Big Ben", os.path.join(root, "big_ben.jpg")))
    landmarks.append(Landmark("Bosphorus Bridge", os.path.join(root, "boğaziçi_köprüsü.jpg")))
    landmarks.append(Landmark("Casa Loma", os.path.join(root, "casa_loma.jpg")))
    landmarks.append(Landmark("Castel Santangelo", os.path.join(root, "castel_santangelo.jpg")))
    landmarks.append(Landmark("Charlottenburg Palace", os.path.join(root, "charlottenburg_palace.jpg")))
    landmarks.append(Landmark("Colosseum", os.path.join(root, "colosseum.jpg")))
    landmarks.append(Landmark("Dresden Frauenkirche", os.path.join(root, "dresden_frauenkirche.jpg")))
    landmarks.append(Landmark("Eiffel Tower", os.path.join(root, "eiffel_tower.jpg")))
    landmarks.append(Landmark("Galata Tower", os.path.join(root, "galata_kulesi.jpg")))
    landmarks.append(Landmark("Golden Gate", os.path.join(root, "golden_gate.jpg")))
    landmarks.append(Landmark("Independence Hall", os.path.join(root, "independence_hall.jpg")))
    landmarks.append(Landmark("İshak Pasha Palace", os.path.join(root, "ishak_pasa_sarayi.jpg")))
    landmarks.append(Landmark("Maiden's Tower", os.path.join(root, "kiz_kulesi.jpg")))
    landmarks.append(Landmark("Leaning Tower of Pisa", os.path.join(root, "leaning_tower_of_pisa.jpg")))
    landmarks.append(Landmark("Milan Cathedral", os.path.join(root, "milan_cathedral.jpg")))
    landmarks.append(Landmark("Notre Dame Cathedral", os.path.join(root, "notre_dame_cathedral.jpg")))
    landmarks.append(Landmark("Pamukkale", os.path.join(root, "pamukkale.jpg")))
    landmarks.append(Landmark("Pantheon", os.path.join(root, "pantheon.jpg")))
    landmarks.append(Landmark("Pyramids", os.path.join(root, "pyramids.jpg")))
    landmarks.append(Landmark("Reichstag", os.path.join(root, "reichstag.jpg")))
    landmarks.append(Landmark("Sacre Coeur Basilica", os.path.join(root, "sacre_coeur_basilica.jpg")))
    landmarks.append(Landmark("Savior on Blood Church", os.path.join(root, "savior_on_blood_church.jpg")))
    landmarks.append(Landmark("Statue of Liberty", os.path.join(root, "statue_of_liberty.jpg")))
    landmarks.append(Landmark("Stephansdom", os.path.join(root, "stephansdom.jpg")))
    landmarks.append(Landmark("St. Basil Cathedral", os.path.join(root, "st_basil_cathedral.jpg")))
    landmarks.append(Landmark("St. Louis Cathedral", os.path.join(root, "st_louis_cathedral.jpg")))
    landmarks.append(Landmark("St. Paul Cathedral", os.path.join(root, "st_paul_cathedral.jpg")))
    landmarks.append(Landmark("Süleymaniye Mosque", os.path.join(root, "suleymaniye.jpg")))
    landmarks.append(Landmark("Blue Mosque", os.path.join(root, "sultanahmet.jpg")))
    landmarks.append(Landmark("Taj Mahal", os.path.join(root, "taj_mahal.jpg")))
    landmarks.append(Landmark("Trevi Fountain", os.path.join(root, "trevi_fountain.jpg")))
    landmarks.append(Landmark("Tyn Church", os.path.join(root, "tyn_church.jpg")))
    landmarks.append(Landmark("Westminster Abbey", os.path.join(root, "westminster_abbey.jpg")))

    return landmarks