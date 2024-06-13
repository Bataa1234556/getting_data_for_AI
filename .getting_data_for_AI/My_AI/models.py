# myapp/models.py

from django.db import models

class Uildverlegch(models.Model):
    FORD = 0
    HONDA = 1
    HYUNDAI = 2
    LEXUS = 3
    MERCEDES_BENZ = 4
    MITSUBISHI = 5
    NISSAN = 6
    SUBARU = 7
    TOYOTA = 8

    UILVERLEGCH_CHOICES = [
        (FORD, 'Ford'),
        (HONDA, 'Honda'),
        (HYUNDAI, 'Hyundai'),
        (LEXUS, 'Lexus'),
        (MERCEDES_BENZ, 'Mercedes-Benz'),
        (MITSUBISHI, 'Mitsubishi'),
        (NISSAN, 'Nissan'),
        (SUBARU, 'Subaru'),
        (TOYOTA, 'Toyota'),
    ]

    name = models.PositiveSmallIntegerField(choices=UILVERLEGCH_CHOICES)

    def __str__(self):
        return self.get_name_display()

class Mark(models.Model):
    MARK_CHOICES = [
        (0, '3000GT'),
        (1, '4Runner'),
        (2, 'A-Class'),
        (3, 'Accent'),
        (4, 'Accord'),
        (5, 'Advan'),
        (6, 'Allion'),
        (7, 'Alphard'),
        (8, 'Altima'),
        (9, 'Aqua'),
        (10, 'Auris'),
        (11, 'Avante'),
        (12, 'Azera'),
        (13, 'B-Class'),
        (14, 'BRZ'),
        (15, 'Belta'),
        (16, 'Bluebird'),
        (17, 'Brevis'),
        (18, 'C-Class'),
        (19, 'C-HR'),
        (20, 'CAPA'),
        (21, 'CL-Class'),
        (22, 'CLA-Class'),
        (23, 'CLK-Class'),
        (24, 'CLS-Class'),
        (25, 'CR-V'),
        (26, 'CR-Z'),
        (27, 'CT'),
        (28, 'Camry'),
        (29, 'Caravan'),
        (30, 'Challenger'),
        (31, 'Chaser'),
        (32, 'Civic'),
        (33, 'Colt'),
        (34, 'Corolla'),
        (35, 'Crossroad'),
        (36, 'Crown'),
        (37, 'Cube'),
        (38, 'Datsun'),
        (39, 'Delica'),
        (40, 'Dualis/Qashqai'),
        (41, 'E-Class'),
        (42, 'ES'),
        (43, 'Eclipse'),
        (44, 'Elantra'),
        (45, 'Elgrand'),
        (46, 'Equus'),
        (47, 'Escape'),
        (48, 'Estima'),
        (49, 'Everest'),
        (50, 'Exiga'),
        (51, 'Expedition'),
        (52, 'Explorer'),
        (53, 'F-сери'),
        (54, 'FJ Cruiser'),
        (55, 'Fielder'),
        (56, 'Fit'),
        (57, 'Forester'),
        (58, 'Fortuner'),
        (59, 'Fuga'),
        (60, 'G-Class'),
        (61, 'GL-Class'),
        (62, 'GLA-Class'),
        (63, 'GLK-Class'),
        (64, 'GS'),
        (65, 'GX'),
        (66, 'Genesis'),
        (67, 'Grace'),
        (68, 'Grandeur'),
        (69, 'HR-V'),
        (70, 'HS'),
        (71, 'Harrier'),
        (72, 'Hiace'),
        (73, 'Highlander'),
        (74, 'Hilux'),
        (75, 'IS'),
        (76, 'Impreza'),
        (77, 'Insight'),
        (78, 'Isis'),
        (79, 'Ist'),
        (80, 'Juke'),
        (81, 'Kluger'),
        (82, 'L200'),
        (83, 'LS'),
        (84, 'LX'),
        (85, 'Lancer'),
        (86, 'Land Cruiser 100'),
        (87, 'Land Cruiser 105'),
        (88, 'Land Cruiser 200'),
        (89, 'Land Cruiser 300'),
        (90, 'Land Cruiser 70'),
        (91, 'Land Cruiser 77'),
        (92, 'Land Cruiser 80'),
        (93, 'Land Cruiser Prado'),
        (94, 'Latio'),
        (95, 'Leaf'),
        (96, 'Legacy'),
        (97, 'Liberty'),
        (98, 'Logo'),
        (99, 'M-Class'),
        (100, 'March'),
        (101, 'Mark'),
        (102, 'Maxima'),
        (103, 'Mirage'),
        (104, 'Montero'),
        (105, 'Murano'),
        (106, 'Mustang'),
        (107, 'NV200'),
        (108, 'NX'),
        (109, 'Navara'),
        (110, 'Noah'),
        (111, 'Note'),
        (112, 'Odyssey'),
        (113, 'Outback'),
        (114, 'Outlander'),
        (115, 'Pajero'),
        (116, 'Passo'),
        (117, 'Pathfinder'),
        (118, 'Patrol'),
        (119, 'Pickup'),
        (120, 'Presage'),
        (121, 'Prius 10'),
        (122, 'Prius 20'),
        (123, 'Prius 30'),
        (124, 'Prius 40'),
        (125, 'Prius 50'),
        (126, 'Probox'),
        (127, 'Quest'),
        (128, 'RAV4'),
        (129, 'RVR'),
        (130, 'RX'),
        (131, 'Ractis'),
        (132, 'Ranger'),
        (133, 'Raum'),
        (134, 'Ridgeline'),
        (135, 'Rogue'),
        (136, 'Rumion'),
        (137, 'S-Class'),
        (138, 'SLK-Class'),
        (139, 'Sai'),
        (140, 'Santa-Fe'),
        (141, 'Serena'),
        (142, 'Sienna'),
        (143, 'Skyline'),
        (144, 'Sonata'),
        (145, 'Spacio'),
        (146, 'Sprinter'),
        (147, 'Starex'),
        (148, 'Sunny'),
        (149, 'Tacoma'),
        (150, 'Teana'),
        (151, 'Terracan'),
        (152, 'Tiida'),
        (153, 'Transit'),
        (154, 'Tribeca'),
        (155, 'Tucson'),
        (156, 'Tundra'),
        (157, 'Vanette'),
        (158, 'Vanguard'),
        (159, 'Vellfire'),
        (160, 'Veracruz'),
        (161, 'Verna'),
        (162, 'Vitz'),
        (163, 'Voxy'),
        (164, 'WRX'),
        (165, 'WRX STI'),
        (166, 'Wingroad'),
        (167, 'Wish'),
        (168, 'X-Trail'),
        (169, 'XT'),
        (170, 'XV Crosstrek'),
        (171, 'bB'),
        (172, 'Бусад'),
    ]

    name = models.PositiveSmallIntegerField(choices=MARK_CHOICES)
    uildverlegch = models.ForeignKey(Uildverlegch, on_delete=models.CASCADE)

    def __str__(self):
        return self.get_name_display()

class MotorBagtaamj(models.Model):
    size = models.CharField(max_length=100)
    mark = models.ForeignKey(Mark, on_delete=models.CASCADE)

    def __str__(self):
        return self.size

class Xrop(models.Model):

    AUTO = 1
    MECHANICH = 0

    XROP_CHOICES = [
        (AUTO, 'Автомат'),
        (MECHANICH, 'Механик')
    ]


    type = models.PositiveSmallIntegerField(choices=XROP_CHOICES)
    motor_bagtaamj = models.ForeignKey(MotorBagtaamj, on_delete=models.CASCADE)



    def __str__(self):
        return self.type

class Joloo(models.Model):

    ZUW = 1
    BURUU = 0

    JOLOO_CHOICES = [
        (ZUW, 'Зөв'),
        (BURUU, 'Буруу')
    ]

    type = models.PositiveSmallIntegerField(choices=JOLOO_CHOICES)
    xrop = models.ForeignKey(Xrop, on_delete=models.CASCADE)

    def __str__(self):
        return self.type

class UildverlesenOn(models.Model):
    year = models.CharField(max_length=100)
    joloo = models.ForeignKey(Joloo, on_delete=models.CASCADE)

    def __str__(self):
        return self.year

class OrjIrsenOn(models.Model):
    year = models.CharField(max_length=100)
    uildverlesen_on = models.ForeignKey(UildverlesenOn, on_delete=models.CASCADE)

    def __str__(self):
        return self.year

class Hutlugch(models.Model):
    BUH_DUGUI_4WD = 0
    URDAA_FWD = 1
    HOINOO_RWD = 2

    HUTLUGCH_CHOICES = [
        (BUH_DUGUI_4WD, 'Бүх дугуй 4WD'),
        (URDAA_FWD, 'Урдаа FWD'),
        (HOINOO_RWD, 'Хойноо RWD'),
    ]

    type = models.PositiveSmallIntegerField(choices=HUTLUGCH_CHOICES)
    orj_irsen_on = models.ForeignKey(OrjIrsenOn, on_delete=models.CASCADE)

    def __str__(self):
        return self.get_type_display()

class YavsanKm(models.Model):
    distance = models.CharField(max_length=100)
    hutlugch = models.ForeignKey(Hutlugch, on_delete=models.CASCADE)

    def __str__(self):
        return self.distance

class Hudulguur(models.Model):
    HUDULGUUR_CHOICES = [
        ('Бензин', 'Бензин'),
        ('Газ', 'Газ'),
        ('Дизель', 'Дизель'),
        ('Хайбрид', 'Хайбрид'),
        ('Цахилгаан', 'Цахилгаан'),
    ]
    type = models.CharField(max_length=100, choices=HUDULGUUR_CHOICES)

    def __str__(self):
        return self.type