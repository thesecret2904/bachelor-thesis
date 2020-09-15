import numpy as np


def fourier_transform(t, func, norm: float = np.sqrt(2 * np.pi), shift=True):
    dt = t[1] - t[0]
    if callable(func):
        values = np.array(list(map(func, t)))
    else:
        values = func

    # do the fast fourier transform and order the values with rising frequencies
    transform = np.fft.fft(values) * dt / norm

    # calculate the frequencies and again order them
    freqs = np.fft.fftfreq(len(t), dt / (2 * np.pi))
    # do the phase shift to compensate the time not starting at zero
    transform = transform * np.exp(1j * freqs * t[0])

    if shift:
        transform = np.fft.fftshift(transform)
        freqs = np.fft.fftshift(freqs)

    # return the frequency domain and the fourier transform at those values
    return freqs, transform


def inverse_fourier_transform(t, func, norm: float = np.sqrt(2 * np.pi)):
    dt = t[1] - t[0]
    if callable(func):
        values = np.array(list(map(func, t)))
    else:
        values = func

    # do the fast inverse fourier transform and order the values with rising times
    transform = np.fft.fftshift(np.fft.ifft(values)) * dt / norm * len(values)

    # calculate the times and again order them
    freqs = np.fft.fftshift(np.fft.fftfreq(len(t), dt / (2 * np.pi)))
    # do the phase shift to compensate the time not starting at zero
    transform = transform * np.exp(-1j * freqs * t[0])

    return freqs, transform


def main():
    import matplotlib.pyplot as plt
    T = 10.
    t = np.linspace(-T, T, 10000)

    def gaussian(t):
        return np.exp(-np.square(t) / 2) / np.sqrt(2 * np.pi)

    a = 2.5

    # function which should be transformed
    def f(t):
        return 1 if -a <= t <= a else 0

    # analytical fourier transform
    def hat_f(t):
        return np.sqrt(2 / np.pi) * np.sin(a * t) / t

    freqs, transform = fourier_transform(t, f)
    plt.plot(freqs, hat_f(freqs), label='analytisch')
    plt.plot(freqs, transform, label='fft')
    plt.xlim((-50, 50))
    plt.legend()
    plt.show()

    plt.plot(t, gaussian(t))
    freqs, inverse = inverse_fourier_transform(t, gaussian(t))
    plt.plot(freqs, inverse)
    plt.xlim((-5,5))
    plt.show()

    f = 1
    freqs, transform = fourier_transform(t, t)
    plt.plot(freqs, transform.real, label='real')
    plt.plot(freqs, transform.imag, label='imag')
    plt.legend()
    plt.xlim((-5, 5))
    plt.show()



if __name__ == '__main__':
    main()
