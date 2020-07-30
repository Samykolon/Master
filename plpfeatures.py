import numpy
import numpy.matlib
import scipy
import pandas
import scipy.stats as stats
from scipy.signal import lfilter
from scipy.fftpack.realtransforms import dct

PARAM_TYPE = numpy.float32

def rasta_filt(x):
    """Apply RASTA filtering to the input signal.

    :param x: the input audio signal to filter.
        cols of x = critical bands, rows of x = frame
        same for y but after filtering
        default filter is single pole at 0.94
    """
    x = x.T
    numerator = numpy.arange(.2, -.3, -.1)
    denominator = numpy.array([1, -0.94])

    # Initialize the state.  This avoids a big spike at the beginning
    # resulting from the dc offset level in each band.
    # (this is effectively what rasta/rasta_filt.c does).
    # Because Matlab uses a DF2Trans implementation, we have to
    # specify the FIR part to get the state right (but not the IIR part)
    y = numpy.zeros(x.shape)
    zf = numpy.zeros((x.shape[0], 4))
    for i in range(y.shape[0]):
        y[i, :4], zf[i, :4] = lfilter(numerator, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])

    # .. but don't keep any of these values, just output zero at the beginning
    y = numpy.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numerator, denominator, x[i, 4:], axis=-1, zi=zf[i, :])[0]

    return y.T

def hz2mel(f, htk=True):
    """Convert an array of frequency in Hz into mel.

    :param f: frequency to convert

    :return: the equivalence on the mel scale.
    """
    if htk:
        return 2595 * numpy.log10(1 + f / 700.)
    else:
        f = numpy.array(f)

        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        # Mel fn to match Slaney's Auditory Toolbox mfcc.m
        f_0 = 0.
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = f < brkfrq

        z = numpy.zeros_like(f)
        # fill in parts separately
        z[linpts] = (f[linpts] - f_0) / f_sp
        z[~linpts] = brkpt + (numpy.log(f[~linpts] / brkfrq)) / numpy.log(logstep)

        if z.shape == (1,):
            return z[0]
        else:
            return z

def fft2barkmx(n_fft, fs, nfilts=0, width=1., minfreq=0., maxfreq=8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per bark), and width is the constant width of each
    band in Bark (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(n_fft,fs) * abs(fft(xincols, n_fft));
    2004-09-05  dpwe@ee.columbia.edu  based on rastamat/audspec.m

    :param n_fft: the source FFT size at sampling rate fs
    :param fs: sampling rate
    :param nfilts: number of output bands required
    :param width: constant width of each band in Bark (default 1)
    :param minfreq:
    :param maxfreq:
    :return: a matrix of weights to combine FFT bins into Bark bins
    """
    maxfreq = min(maxfreq, fs / 2.)

    min_bark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - min_bark

    if nfilts == 0:
        nfilts = numpy.ceil(nyqbark) + 1

    wts = numpy.zeros((nfilts, n_fft))

    # bark per filt
    step_barks = nyqbark / (nfilts - 1)

    # Frequency of each FFT bin in Bark
    binbarks = hz2bark(numpy.arange(n_fft / 2 + 1) * fs / n_fft)

    for i in range(nfilts):
        f_bark_mid = min_bark + i * step_barks
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = (binbarks - f_bark_mid - 0.5)
        hif = (binbarks - f_bark_mid + 0.5)
        wts[i, :n_fft // 2 + 1] = 10 ** (numpy.minimum(numpy.zeros_like(hif), numpy.minimum(hif, -2.5 * lof) / width))

    return wts

def mel2hz(z, htk=True):
    """Convert an array of mel values in Hz.

    :param m: ndarray of frequencies to convert in Hz.

    :return: the equivalent values in Hertz.
    """
    if htk:
        return 700. * (10**(z / 2595.) - 1)
    else:
        z = numpy.array(z, dtype=float)
        f_0 = 0
        f_sp = 200. / 3.
        brkfrq = 1000.
        brkpt  = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)

        linpts = (z < brkpt)

        f = numpy.zeros_like(z)

        # fill in parts separately
        f[linpts] = f_0 + f_sp * z[linpts]
        f[~linpts] = brkfrq * numpy.exp(numpy.log(logstep) * (z[~linpts] - brkpt))

        if f.shape == (1,):
            return f[0]
        else:
            return f

def dolpc(x, model_order=8):
    """
    compute autoregressive model from spectral magnitude samples

    :param x:
    :param model_order:
    :return:
    """
    nframes, nbands = x.shape

    r = numpy.real(numpy.fft.ifft(numpy.hstack((x,x[:,numpy.arange(nbands-2,0,-1)]))))

    # First half only
    r = r[:, :nbands]

    # Find LPC coeffs by Levinson-Durbin recursion
    y_lpc = numpy.ones((r.shape[0], model_order + 1))

    for ff in range(r.shape[0]):
        y_lpc[ff, 1:], e, _ = levinson(r[ff, :-1].T, order=model_order, allow_singularity=True)
        # Normalize each poly by gain
        y_lpc[ff, :] /= e

    return y_lpc

def bark2hz(z):
    """
    Converts frequencies Bark to Hertz (Hz)

    :param z:
    :return:
    """
    return 600. * numpy.sinh(z / 6.)

def postaud(x, fmax, fbtype='bark', broaden=0):
    """
    do loudness equalization and cube root compression

    :param x:
    :param fmax:
    :param fbtype:
    :param broaden:
    :return:
    """
    nframes, nbands = x.shape

    # Include frequency points at extremes, discard later
    nfpts = nbands + 2 * broaden

    if fbtype == 'bark':
        bandcfhz = bark2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2bark(fmax), num=nfpts))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':
        bandcfhz = mel2hz(numpy.linspace(0, hz2mel(fmax,1), num=nfpts),1)
    else:
        print('unknown fbtype {}'.format(fbtype))

    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[broaden:(nfpts - broaden)]

    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz ** 2
    ftmp = fsq + 1.6e5
    eql = ((fsq / ftmp) ** 2) * ((fsq + 1.44e6) / (fsq + 9.61e6))

    # weight the critical bands
    z = numpy.matlib.repmat(eql.T,nframes,1) * x

    # cube root compress
    z = z ** .33

    # replicate first and last band (because they are unreliable as calculated)
    if broaden == 1:
      y = z[:, numpy.hstack((0,numpy.arange(nbands), nbands - 1))]
    else:
      y = z[:, numpy.hstack((1,numpy.arange(1, nbands - 1), nbands - 2))]

    return y, eql

def fft2melmx(n_fft,
              fs=8000,
              nfilts=0,
              width=1.,
              minfreq=0,
              maxfreq=4000,
              htkmel=False,
              constamp=False):
    """
    Generate a matrix of weights to combine FFT bins into Mel
    bins.  n_fft defines the source FFT size at sampling rate fs.
    Optional nfilts specifies the number of output bands required
    (else one per "mel/width"), and width is the constant width of each
    band relative to standard Mel (default 1).
    While wts has n_fft columns, the second half are all zero.
    Hence, Mel spectrum is fft2melmx(n_fft,fs)*abs(fft(xincols,n_fft));
    minfreq is the frequency (in Hz) of the lowest band edge;
    default is 0, but 133.33 is a common standard (to skip LF).
    maxfreq is frequency in Hz of upper edge; default fs/2.
    You can exactly duplicate the mel matrix in Slaney's mfcc.m
    as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    htkmel=1 means use HTK's version of the mel curve, not Slaney's.
    constamp=1 means make integration windows peak at 1, not sum to 1.
    frqs returns bin center frqs.

    % 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx

    :param n_fft:
    :param fs:
    :param nfilts:
    :param width:
    :param minfreq:
    :param maxfreq:
    :param htkmel:
    :param constamp:
    :return:
    """
    maxfreq = min(maxfreq, fs / 2.)

    if nfilts == 0:
        nfilts = numpy.ceil(hz2mel(maxfreq, htkmel) / 2.)

    wts = numpy.zeros((nfilts, n_fft))

    # Center freqs of each FFT bin
    fftfrqs = numpy.arange(n_fft / 2 + 1) / n_fft * fs

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfreq, htkmel)
    maxmel = hz2mel(maxfreq, htkmel)
    binfrqs = mel2hz(minmel +  numpy.arange(nfilts + 2) / (nfilts + 1) * (maxmel - minmel), htkmel)

    for i in range(nfilts):
        _fs = binfrqs[i + numpy.arange(3, dtype=int)]
        # scale by width
        _fs = _fs[1] + width * (_fs - _fs[1])
        # lower and upper slopes for all bins
        loslope = (fftfrqs - _fs[0]) / (_fs[1] - __fs[0])
        hislope = (_fs[2] - fftfrqs)/(_fs[2] - _fs[1])

        wts[i, 1 + numpy.arange(n_fft//2 + 1)] =numpy.maximum(numpy.zeros_like(loslope),numpy.minimum(loslope, hislope))

    if not constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = numpy.dot(numpy.diag(2. / (binfrqs[2 + numpy.arange(nfilts)] - binfrqs[numpy.arange(nfilts)])) , wts)

    # Make sure 2nd half of FFT is zero
    wts[:, n_fft // 2 + 1: n_fft] = 0

    return wts, binfrqs

def audspec(power_spectrum,
            fs=16000,
            nfilts=None,
            fbtype='bark',
            minfreq=0,
            maxfreq=8000,
            sumpower=True,
            bwidth=1.):
    """

    :param power_spectrum:
    :param fs:
    :param nfilts:
    :param fbtype:
    :param minfreq:
    :param maxfreq:
    :param sumpower:
    :param bwidth:
    :return:
    """
    if nfilts is None:
        nfilts = int(numpy.ceil(hz2bark(fs / 2)) + 1)

    if not fs == 16000:
        maxfreq = min(fs / 2, maxfreq)

    nframes, nfreqs = power_spectrum.shape
    n_fft = (nfreqs -1 ) * 2

    if fbtype == 'bark':
        wts = fft2barkmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'mel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq)
    elif fbtype == 'htkmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, True)
    elif fbtype == 'fcmel':
        wts = fft2melmx(n_fft, fs, nfilts, bwidth, minfreq, maxfreq, True, False)
    else:
        print('fbtype {} not recognized'.format(fbtype))

    wts = wts[:, :nfreqs]

    if sumpower:
        audio_spectrum = power_spectrum.dot(wts.T)
    else:
        audio_spectrum = numpy.dot(numpy.sqrt(power_spectrum), wts.T)**2

    return audio_spectrum, wts

def framing(sig, win_size, win_shift=1, context=(0, 0), pad='zeros'):
    """
    :param sig: input signal, can be mono or multi dimensional
    :param win_size: size of the window in term of samples
    :param win_shift: shift of the sliding window in terme of samples
    :param context: tuple of left and right context
    :param pad: can be zeros or edge
    """
    dsize = sig.dtype.itemsize
    if sig.ndim == 1:
        sig = sig[:, numpy.newaxis]
    # Manage padding
    c = (context, ) + (sig.ndim - 1) * ((0, 0), )
    _win_size = win_size + sum(context)
    shape = (int((sig.shape[0] - win_size) / win_shift) + 1, 1, _win_size, sig.shape[1])
    strides = tuple(map(lambda x: x * dsize, [win_shift * sig.shape[1], 1, sig.shape[1], 1]))
    if pad == 'zeros':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'constant', constant_values=(0,)),
                                                  shape=shape,
                                                  strides=strides).squeeze()
    elif pad == 'edge':
        return numpy.lib.stride_tricks.as_strided(numpy.lib.pad(sig, c, 'edge'),
                                                  shape=shape,
                                                  strides=strides).squeeze()

def power_spectrum(input_sig,
                   fs=48000,
                   win_time=0.8,
                   shift=0.1,
                   prefac=0.0):
    """
    Compute the power spectrum of the signal.
    :param input_sig:
    :param fs:
    :param win_time:
    :param shift:
    :param prefac:
    :return:
    """
    window_length = int(round(win_time * fs))
    overlap = window_length - int(shift * fs)
    framed = framing(input_sig, window_length, win_shift=window_length-overlap).copy()

    l = framed.shape[0]
    n_fft = 2 ** int(numpy.ceil(numpy.log2(window_length)))
    # Windowing has been changed to hanning which is supposed to have less noisy sidelobes
    # ham = numpy.hamming(window_length)
    window = numpy.hanning(window_length)

    spec = numpy.ones((l, int(n_fft / 2) + 1), dtype=PARAM_TYPE)
    dec = 500000
    start = 0
    stop = min(dec, l)
    while start < l:
        ahan = framed[start:stop, :] * window
        mag = numpy.fft.rfft(ahan, n_fft, axis=-1)
        spec[start:stop, :] = mag.real**2 + mag.imag**2
        start = stop
        stop = min(stop + dec, l)

    return spec

def lpc2cep(a, nout):
    """
    Convert the LPC 'a' coefficients in each column of lpcas
    into frames of cepstra.
    nout is number of cepstra to produce, defaults to size(lpcas,1)
    2003-04-11 dpwe@ee.columbia.edu

    :param a:
    :param nout:
    :return:
    """
    ncol , nin = a.shape

    order = nin - 1

    if nout is None:
        nout = order + 1

    c = numpy.zeros((ncol, nout))

    # First cep is log(Error) from Durbin
    c[:, 0] = -numpy.log(a[:, 0])

    # Renormalize lpc A coeffs
    a /= numpy.tile(a[:, 0][:, None], (1, nin))

    for n in range(1, nout):
        sum = 0
        for m in range(1, n):
            sum += (n - m)  * a[:, m] * c[:, n - m]
        c[:, n] = -(a[:, n] + sum / n)

    return c

def lpc2spec(lpcas, nout=17):
    """
    Convert LPC coeffs back into spectra
    nout is number of freq channels, default 17 (i.e. for 8 kHz)

    :param lpcas:
    :param nout:
    :return:
    """
    [cols, rows] = lpcas.shape
    order = rows - 1

    gg = lpcas[:, 0]
    aa = lpcas / numpy.tile(gg, (rows,1)).T

    # Calculate the actual z-plane polyvals: nout points around unit circle
    zz = numpy.exp((-1j * numpy.pi / (nout - 1)) * numpy.outer(numpy.arange(nout).T,  numpy.arange(order + 1)))

    # Actual polyvals, in power (mag^2)
    features = ( 1./numpy.abs(aa.dot(zz.T))**2) / numpy.tile(gg, (nout, 1)).T

    F = numpy.zeros((cols, rows-1))
    M = numpy.zeros((cols, rows-1))

    for c in range(cols):
        aaa = aa[c, :]
        rr = numpy.roots(aaa)
        ff = numpy.angle(rr.T)
        zz = numpy.exp(1j * numpy.outer(ff, numpy.arange(len(aaa))))
        mags = numpy.sqrt(((1./numpy.abs(zz.dot(aaa)))**2)/gg[c])
        ix = numpy.argsort(ff)
        keep = ff[ix] > 0
        ix = ix[keep]
        F[c, numpy.arange(len(ix))] = ff[ix]
        M[c, numpy.arange(len(ix))] = mags[ix]

    F = F[:, F.sum(axis=0) != 0]
    M = M[:, M.sum(axis=0) != 0]

    return features, F, M

def spec2cep(spec, ncep=13, type=2):
    """
    Calculate cepstra from spectral samples (in columns of spec)
    Return ncep cepstral rows (defaults to 9)
    This one does type II dct, or type I if type is specified as 1
    dctm returns the DCT matrix that spec was multiplied by to give cep.

    :param spec:
    :param ncep:
    :param type:
    :return:
    """
    nrow, ncol = spec.shape

    # Make the DCT matrix
    dctm = numpy.zeros(ncep, nrow);
    #if type == 2 || type == 3
    #    # this is the orthogonal one, the one you want
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:2:(2*nrow-1)]/(2*nrow)*pi) * sqrt(2/nrow);

    #    if type == 2
    #        # make it unitary! (but not for HTK type 3)
    #        dctm(1,:) = dctm(1,:)/sqrt(2);

    #elif type == 4:  # type 1 with implicit repeating of first, last bins
    #    """
    #    Deep in the heart of the rasta/feacalc code, there is the logic
    #    that the first and last auditory bands extend beyond the edge of
    #    the actual spectra, and they are thus copied from their neighbors.
    #    Normally, we just ignore those bands and take the 19 in the middle,
    #    but when feacalc calculates mfccs, it actually takes the cepstrum
    #    over the spectrum *including* the repeated bins at each end.
    #    Here, we simulate 'repeating' the bins and an nrow+2-length
    #    spectrum by adding in extra DCT weight to the first and last
    #    bins.
    #    """
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[1:nrow]/(nrow+1)*pi) * 2;
    #        # Add in edge points at ends (includes fixup scale)
    #        dctm(i,1) = dctm(i,1) + 1;
    #        dctm(i,nrow) = dctm(i,nrow) + ((-1)^(i-1));

    #   dctm = dctm / (2*(nrow+1));
    #else % dpwe type 1 - same as old spec2cep that expanded & used fft
    #    for i = 1:ncep
    #        dctm(i,:) = cos((i-1)*[0:(nrow-1)]/(nrow-1)*pi) * 2 / (2*(nrow-1));
    #    dctm(:,[1 nrow]) = dctm(:, [1 nrow])/2;

    #cep = dctm*log(spec);
    return None, None, None


def lifter(x, lift=0.6, invs=False):
    """
    Apply lifter to matrix of cepstra (one per column)
    lift = exponent of x i^n liftering
    or, as a negative integer, the length of HTK-style sin-curve liftering.
    If inverse == 1 (default 0), undo the liftering.

    :param x:
    :param lift:
    :param invs:
    :return:
    """
    nfrm , ncep = x.shape

    if lift == 0:
        y = x
    else:
        if lift > 0:
            if lift > 10:
                print('Unlikely lift exponent of {} did you mean -ve?'.format(lift))
            liftwts = numpy.hstack((1, numpy.arange(1, ncep)**lift))

        elif lift < 0:
        # Hack to support HTK liftering
            L = float(-lift)
            if (L != numpy.round(L)):
                print('HTK liftering value {} must be integer'.format(L))

            liftwts = numpy.hstack((1, 1 + L/2*numpy.sin(numpy.arange(1, ncep) * numpy.pi / L)))

        if invs:
            liftwts = 1 / liftwts

        y = x.dot(numpy.diag(liftwts))

    return y

def hz2bark(f):
    """
    Convert frequencies (Hertz) to Bark frequencies

    :param f: the input frequency
    :return:
    """
    return 6. * numpy.arcsinh(f / 600.)

def plp(input_sig,
         nwin=0.8,
         fs=48000,
         plp_order=13,
         shift=0.1,
         get_spec=False,
         get_mspec=False,
         prefac=0.0,
         rasta=False):
    """
    output is matrix of features, row = feature, col = frame
    % fs is sampling rate of samples, defaults to 8000
    % dorasta defaults to 1; if 0, just calculate PLP
    % modelorder is order of PLP model, defaults to 8.  0 -> no PLP
    :param input_sig:
    :param fs: sampling rate of samples default is 8000
    :param rasta: default is True, if False, juste compute PLP
    :param model_order: order of the PLP model, default is 8, 0 means no PLP
    :return: matrix of features, row = features, column are frames
    """
    plp_order -= 1

    # first compute power spectrum
    powspec = power_spectrum(input_sig, fs, nwin, shift, prefac)

    # next group to critical bands
    audio_spectrum = audspec(powspec, fs)[0]
    nbands = audio_spectrum.shape[0]

    if rasta:
        # put in log domain
        nl_aspectrum = numpy.log(audio_spectrum)

        #  next do rasta filtering
        ras_nl_aspectrum = rasta_filt(nl_aspectrum)

        # do inverse log
        audio_spectrum = numpy.exp(ras_nl_aspectrum)

    # do final auditory compressions
    post_spectrum = postaud(audio_spectrum, fs / 2.)[0]

    if plp_order > 0:

        # LPC analysis
        lpcas = dolpc(post_spectrum, plp_order)

        # convert lpc to cepstra
        cepstra = lpc2cep(lpcas, plp_order + 1)

        # .. or to spectra
        spectra, F, M = lpc2spec(lpcas, nbands)

    else:

        # No LPC smoothing of spectrum
        spectra = post_spectrum
        cepstra = spec2cep(spectra)

    cepstra = lifter(cepstra, 0.6)

    lst = list()
    lst.append(cepstra)

    if get_spec:
        lst.append(powspec)
    else:
        lst.append(None)
        del powspec
    if get_mspec:
        lst.append(post_spectrum)
    else:
        lst.append(None)
        del post_spectrum

    return lst

def levinson(r, order=None, allow_singularity=False):
    r"""Levinson-Durbin recursion.

    Find the coefficients of a length(r)-1 order autoregressive linear process

    :param r: autocorrelation sequence of length N + 1 (first element being the zero-lag autocorrelation)
    :param order: requested order of the autoregressive coefficients. default is N.
    :param allow_singularity: false by default. Other implementations may be True (e.g., octave)

    :return:
        * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
        * the prediction errors
        * the `N` reflections coefficients values

    This algorithm solves the set of complex linear simultaneous equations
    using Levinson algorithm.

    .. math::

        \bold{T}_M \left( \begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
        \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)

    where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
    :math:`T_0, T_1, \dots ,T_M`.

    .. note:: Solving this equations by Gaussian elimination would
        require :math:`M^3` operations whereas the levinson algorithm
        requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.

    This is equivalent to solve the following symmetric Toeplitz system of
    linear equations

    .. math::

        \left( \begin{array}{cccc}
        r_1 & r_2^* & \dots & r_{n}^*\\
        r_2 & r_1^* & \dots & r_{n-1}^*\\
        \dots & \dots & \dots & \dots\\
        r_n & \dots & r_2 & r_1 \end{array} \right)
        \left( \begin{array}{cccc}
        a_2\\
        a_3 \\
        \dots \\
        a_{N+1}  \end{array} \right)
        =
        \left( \begin{array}{cccc}
        -r_2\\
        -r_3 \\
        \dots \\
        -r_{N+1}  \end{array} \right)

    where :math:`r = (r_1  ... r_{N+1})` is the input autocorrelation vector, and
    :math:`r_i^*` denotes the complex conjugate of :math:`r_i`. The input r is typically
    a vector of autocorrelation coefficients where lag 0 is the first
    element :math:`r_1`.


    .. doctest::

        >>> import numpy; from spectrum import LEVINSON
        >>> T = numpy.array([3., -2+0.5j, .7-1j])
        >>> a, e, k = LEVINSON(T)

    """
    #from numpy import isrealobj
    T0  = numpy.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, 'order must be less than size of the input data'
        M = order

    realdata = numpy.isrealobj(r)
    if realdata is True:
        A = numpy.zeros(M, dtype=float)
        ref = numpy.zeros(M, dtype=float)
    else:
        A = numpy.zeros(M, dtype=complex)
        ref = numpy.zeros(M, dtype=complex)

    P = T0

    for k in range(M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            #save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k-j-1]
            temp = -save / P
        if realdata:
            P = P * (1. - temp**2.)
        else:
            P = P * (1. - (temp.real**2+temp.imag**2))

        if (P <= 0).any() and allow_singularity==False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k+1)//2
        if realdata is True:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp*save
        else:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref
