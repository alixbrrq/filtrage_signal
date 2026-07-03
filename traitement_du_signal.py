import numpy as np
import matplotlib.pyplot as plt

# 1. Signal générique

class Signal:
    """Représente un signal échantillonné quelconque."""

    def __init__(self, Ne, fe):
        self.Ne = Ne  # nombre d'échantillons
        self.fe = fe  # fréquence d'échantillonnage (Hz)
        self.x = np.zeros(Ne)  # vecteur des échantillons

    def tracer(self): # tracer le signal avec matplotlib
        t = np.arange(len(self.x)) / self.fe
        plt.plot(t, self.x)
        plt.xlabel("Temps (s)")
        plt.ylabel("Amplitude")
        plt.title("Signal")
        plt.grid()
        plt.show()

    def __repr__(self):
        return f"{self.__class__.__name__}(Ne={self.Ne}, fe={self.fe} Hz)"


# 2. Signaux périodiques

class SignalPeriodique(Signal):

    def __init__(self, Ne, fe=8000, amplitude=1.0, frequence=1.0):
        super().__init__(Ne, fe)
        self.amplitude = amplitude
        self.frequence = frequence

        n = np.arange(Ne)
        phase = 2 * np.pi * self.frequence * n / self.fe
        self.x = amplitude * self._forme(phase)

    def _forme(self, phase):
        raise NotImplementedError


class SignalSinus(SignalPeriodique):
    def _forme(self, phase):
        return np.sin(phase)


class SignalCarre(SignalPeriodique):
    def _forme(self, phase):
        return np.sign(np.sin(phase))


class SignalTriangle(SignalPeriodique):
    def _forme(self, phase):
        # rampe triangulaire normalisée entre -1 et 1
        return 2 / np.pi * np.arcsin(np.sin(phase))


# 3. Filtres

class Filtre:

    def __init__(self, fe):
        self.fe = fe

    def reponse_impulsionnelle(self, n):
        raise NotImplementedError

    def filtrer(self, data):
        n = np.arange(len(data))
        h = self.reponse_impulsionnelle(n)
        return np.convolve(data, h, mode="same")

    def __repr__(self):
        return f"{self.__class__.__name__}(fe={self.fe}Hz)"


class FiltreMoyenneur(Filtre):

    def __init__(self, fe, N=5):
        super().__init__(fe)
        self.N = N

    def reponse_impulsionnelle(self, n):
        h = np.zeros_like(n, dtype=float)
        h[n < self.N] = 1.0 / self.N
        return h


class FiltrePasseBasExponentiel(Filtre):

    def __init__(self, fe, alpha=0.2):
        super().__init__(fe)
        self.alpha = alpha

    def reponse_impulsionnelle(self, n):
        return self.alpha * (1 - self.alpha) ** n

# 4. Démonstration
def main():

    # signal sinusoïdal
    signal = SignalSinus(Ne=4000, fe=2000, amplitude=1.0, frequence=5)

    print(signal)

    # bruit
    bruit = np.random.normal(0, 0.3, size=signal.x.shape)
    signal_bruite = Signal(signal.Ne, signal.fe)
    signal_bruite.x = signal.x + bruit

    # filtres
    filtre_moy = FiltreMoyenneur(fe=signal.fe, N=9)
    filtre_rc = FiltrePasseBasExponentiel(fe=signal.fe, alpha=0.25)

    # filtrage
    signal_moy = Signal(signal.Ne, signal.fe)
    signal_moy.x = filtre_moy.filtrer(signal_bruite.x)

    signal_rc = Signal(signal.Ne, signal.fe)
    signal_rc.x = filtre_rc.filtrer(signal_bruite.x)

    # stats
    print(f"Écart-type bruité : {np.std(signal_bruite.x):.3f}")
    print(f"Après moyenneur   : {np.std(signal_moy.x):.3f}")
    print(f"Après RC          : {np.std(signal_rc.x):.3f}")


    signal_bruite.tracer()
    signal_moy.tracer()
    signal_rc.tracer()
    
    return signal_bruite, signal_moy, signal_rc


if __name__ == "__main__":
    main()