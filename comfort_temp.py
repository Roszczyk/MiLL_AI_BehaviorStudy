import pythermalcomfort.models as ptc

# Przykładowe wartości wskaźnika met w zależności od aktywności:

# MET:
# 0.8 – siedzenie, odpoczynek (np. praca biurowa, oglądanie telewizji).
# 1.2 – lekkie czynności (np. praca w pozycji stojącej, praca przy komputerze).
# 1.5 – spacer z prędkością 4 km/h (np. wolny marsz).
# 2.0 – lekkie prace fizyczne (np. prace domowe, chodzenie po schodach).
# 3.0 – szybki marsz, praca fizyczna (np. chodzenie z prędkością 6 km/h, intensywne sprzątanie).
# 4.0 – bieganie z prędkością 8 km/h (np. trucht).

# CLO:
# 0.0 – brak ubrań lub bardzo lekkie ubrania (np. kąpielówki, strój plażowy).
# 0.5 – lekkie ubrania (np. letnie ubrania, koszulka z krótkim rękawem, krótkie spodenki).
# 1.0 – standardowe ubrania (np. długie spodnie, koszula, lekka marynarka lub sweter).
# 1.5 – cięższe ubrania (np. zimowy płaszcz, ciepły sweter, szalik).
# 2.0 – bardzo ciepłe ubrania (np. kurtka puchowa, grube spodnie, dodatkowe warstwy ubrań, odzież zimowa).
# 3.0 – ekstremalne warunki (np. odzież na bardzo zimne warunki, kombinezon narciarski, odzież puchowa).


print(ptc.set_tmp(tdb = 22,
            tr = 22,
            v=0,
            rh=60,
            met=1.2,
            clo=0.5))