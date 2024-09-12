import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models


def mkidr(l, R400=25):
    return R400 * 400 / l


def osep(l, R400=25, c=1.33):
    return c * l ** 2 / R400 / 400


fsr = lambda l, m: l / m


def res(l, owid, npix=2048, nspot=3):
    return l / (nspot * owid / npix)


def lim_order(R0=15, l0=400, lf=800, c=1.22, debug=False):
    const = c * lf / R0 / l0
    m = np.arange(2, 150)
    x = 1 / 2 / m + 1
    y = 1 - m / (m + 1)
    z = 1 / 2 / (m + 1) + 1
    mconst = x * y / z
    if debug:
        print(const)
        print(m)
        print(mconst)
        plt.axhline(const, linewidth=.5, color='k')
        plt.plot(m, mconst)
        plt.show()
    return m[mconst >= const].max()


l = 600
nspot = 2.5
r400 = np.linspace(10, 25, 100, 10)

owid = osep(l, R400=r400)

npix = np.array([np.linspace(0, 2048 * 4, 10)]).T

r = l / (nspot * owid / npix) / 1000

#  For Poster


plt.figure(figsize=(6, 3))
x = np.linspace(300, 900, 1000)
m0 = 5
R400 = 15
l0 = 800 / (1 + .5 / m0)
gaussians = []
lmax = 800
lmin = 400
m = 2
l = 1000
while l > 200:
    l = m0 * l0 / m
    sig = osep(l, R400=R400, c=1) / 2.348
    amp = 1 / sig / np.sqrt(2 * np.pi)
    g = models.Gaussian1D(amplitude=amp, mean=l, stddev=sig)
    c = 'grey'
    if lmin < l < lmax:
        gaussians.append(g)
        c = None
    plt.plot(x, g(x), color=c, linewidth=.9)
    m += 1
allg = gaussians[0]
for m in gaussians[1:]:
    allg = allg + m
plt.plot(x, allg(x), 'k', linewidth=1.2)
plt.ylabel('Normalized Flux')
plt.xlabel('Wavelength (nm)')
plt.axvline(lmin, color='k', linewidth=.8)
plt.axvline(lmax, color='k', linewidth=.8)
plt.gca().set_yticklabels([])
plt.xlim(375, 850)
plt.title('$R_{MKID, 400 nm}$=' + f'{R400} m={m0}')
plt.tight_layout()
plt.savefig('pixel_echellogram.eps')
plt.show()


def maxm(l, r400, n):
    lc = 800 / (1 + .5 / m0)
    eta = n / r400 / 400
    return np.floor((1 - l * eta) / l / eta)


maxord = np.array([lim_order(R0=r) for r in r400])


def spec_r(m0, npix):
    lc = 800 / (1 + .5 / m0)

    return lc / (fsr(lc, m0) / (npix / nspot))


npix = np.array([np.linspace(0, 2048 * 4, 10)]).T

rgrid = spec_r(*np.meshgrid(maxord, npix))

contours = plt.contour(*(*np.meshgrid(r400, npix), rgrid),
                       [2500, 7500, 15000, 24000, 36000, 48000, 60000, 75000], colors='white')
plt.clabel(contours, inline=True, fontsize=12, fmt='%1.0f')
plt.imshow(rgrid,
           extent=[r400.min(), r400.max(), npix.min(), npix.max()], origin='lower', vmax=100000,
           alpha=1, aspect='auto', interpolation='gaussian')
plt.colorbar().set_label('Spectrograph Resolution')
plt.xlabel('MKID Resolution')
plt.ylabel('Array Length')
plt.title('Testbench Resolution')
# plt.gca().set_yticklabels([])
plt.plot([15, 40, 50, 70], [2048, 2048, 4096, 4096 + 2048], '*w', markersize=10)
plt.savefig('poster_res.eps')
plt.show()

######
# For Miguel

migr = 10
import matplotlib as mpl

mpl.rcParams['font.size'] = 22
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.linewidth'] = 1.5
# mpl.rcParams['ytick.major.size']=6
# mpl.rcParams['ytick.major.width']=2
# mpl.rcParams['xtick.major.size']=6
# mpl.rcParams['xtick.major.width']=2
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'arial'

g800 = models.Gaussian1D(amplitude=1, mean=800, stddev=osep(800, migr, 1) / 2.355)
g800n = models.Gaussian1D(amplitude=1, mean=700, stddev=osep(700, migr, 1) / 2.355)
lam = np.linspace(350, 900, 1000)
# plt.subplot(2,1,1)
plt.plot(lam, g800(lam), '-.k')
plt.plot(lam, g800n(lam), '--k')

g400 = models.Gaussian1D(amplitude=1, mean=400, stddev=osep(400, migr, 1) / 2.355)
g400n = models.Gaussian1D(amplitude=1, mean=500, stddev=osep(500, migr, 1) / 2.355)

# plt.subplot(2,1,1)
plt.plot(lam, g400(lam), '-.k')
plt.plot(lam, g400n(lam), '--k')
plt.plot(lam, (g800 + g800n + g400n + g400)(lam))

plt.ylabel('Counts')
plt.gca().set_yticklabels([])
plt.text(525, .95, '$R_{{400}}={}$'.format(migr))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Counts')
plt.subplots_adjust(.1, .17, .97, .97)

######3


plt.figure(figsize=(3.5, 6))
g800 = models.Gaussian1D(amplitude=1, mean=800, stddev=osep(800, 25, 1) / 2.355)
g800n = models.Gaussian1D(amplitude=1, mean=4 * 800 / 5, stddev=osep(4 * 800 / 5, 25, 1) / 2.355)
lam = np.linspace(570, 900, 1000)
plt.subplot(2, 1, 1)
plt.plot(lam, g800(lam), '-.k')
plt.plot(lam, g800n(lam), '--k')
plt.plot(lam, (g800 + g800n)(lam))
plt.ylabel('Counts')
plt.gca().set_yticklabels([])
plt.text(660, .95, '$N_{FWHM}=2.1$')
plt.subplot(2, 1, 2)
# lam=np.linspace(630,900,1000)
g800 = models.Gaussian1D(amplitude=1, mean=800, stddev=osep(800, 25, 1) / 2.355)
g800n = models.Gaussian1D(amplitude=1, mean=8 * 800 / 9, stddev=osep(8 * 800 / 9, 25, 1) / 2.355)
plt.plot(lam, g800(lam), '-.k')
plt.plot(lam, g800n(lam), '--k')
plt.plot(lam, (g800 + g800n)(lam))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Counts')
plt.gca().set_yticklabels([])
plt.text(565, .95, '$N_{FWHM}=1.33$')
plt.subplots_adjust(None, None, .95, .95)

# plt.contourf(*(*np.meshgrid(r400, npix), r), 20)


plt.clf()
contours = plt.contour(*(*np.meshgrid(r400, npix), r * 1000), 7, colors='black')
plt.clabel(contours, inline=True, fontsize=12, fmt='%1.0f')
plt.imshow(r, extent=[r400.min(), r400.max(), npix.min(), npix.max()], origin='lower',
           alpha=1, aspect='auto', interpolation='gaussian')
plt.colorbar().set_label('Spectrograph Resolution @ {:.0f} nm (x1000s)'.format(l))
plt.xlabel('MKID Resolution')
plt.ylabel('Pixel Count')

l = 600
nspot = 3
r400 = np.linspace(10, 100, 100)
npix = np.array([np.linspace(0, 2048 * 4, 100)]).T


# maxm = 600/ms

def maxm(l, r400, n):
    eta = n / r400 / 400
    return np.floor((1 - l * eta) / l / eta)


r = lambda m, npix, samp=3: m * npix / samp

plt.clf()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
rgrid = r(*np.meshgrid(maxm(800, r400, 2.1), npix))
contours = plt.contour(*(*np.meshgrid(r400, npix), rgrid),
                       [2000, 7500, 15000, 24000, 36000, 48000], colors='white')
plt.clabel(contours, inline=True, fontsize=12, fmt='%1.0f')
plt.imshow(rgrid,
           extent=[r400.min(), r400.max(), npix.min(), npix.max()], origin='lower',
           vmax=100000,
           alpha=1, aspect='auto', interpolation='gaussian')
# plt.colorbar().set_label('Spectrograph Resolution')
plt.xlabel('MKID Resolution')
plt.ylabel('Pixel Count')
plt.title('2.1 FWHM (1%)')
plt.plot([25, 50, 100], [2048, 4096, 6144], '*w', markersize=10)

plt.subplot(1, 2, 2)
rgrid = r(*np.meshgrid(maxm(800, r400, 1.333), npix))

contours = plt.contour(*(*np.meshgrid(r400, npix), rgrid),
                       [2000, 7500, 15000, 24000, 36000, 48000, 60000, 75000], colors='white')
plt.clabel(contours, inline=True, fontsize=12, fmt='%1.0f')
plt.imshow(rgrid,
           extent=[r400.min(), r400.max(), npix.min(), npix.max()], origin='lower', vmax=100000,
           alpha=1, aspect='auto', interpolation='gaussian')
plt.colorbar().set_label('Spectrograph Resolution')
plt.xlabel('MKID Resolution')
plt.title('1.33 FWHM')
plt.gca().set_yticklabels([])
plt.plot([25, 50, 100], [2048, 4096, 6144], '*w', markersize=10)
plt.subplots_adjust(.1, .1, .95, .95, wspace=.1)


# ----------


def echellogramplot(m_d_iter, title, ylbl=False):
    mr = []
    for m, d in m_d_iter:
        li, fm, lb, fp, lf = d
        ppw = 2048 / (lf - li)
        plt.plot([0, (lf - li) * ppw], [m] * 2, 'k:', linewidth=1)
        plt.plot([(fm - li) * ppw, (fp - li) * ppw], [m] * 2, 'k', linewidth=2)
        # plt.plot([li-lb, lf-lb], [m]*2,'k.',linewidth=2)
        # plt.plot([fm-lb, fp-lb], [m] * 2, 'k', linewidth=2)
        plt.text(30, m + .1, '${:.0f}$'.format(li), size='x-small')
        plt.plot(1024, m, 'k*')
        plt.text(1024, m + .1, 'm={}'.format(m), size='x-small')
        plt.text(2000, m + .1, '${:.0f}$'.format(lf), horizontalalignment='right', size='x-small')
        if mr:
            mr[1] = m
        else:
            mr = [m, 0]

    plt.xlim(0, 2048)
    n = 23
    plt.ylim(np.mean(mr) - n / 2., np.mean(mr) + n / 2.)
    plt.gca().set_yticklabels([])

    plt.xlabel('Pixel')
    # if ylbl: plt.ylabel('Order number')
    plt.title(title)


# 4-7
s = """355.556	377.324	406.349	435.374	457.143
414.815	434.568	474.074	513.580	533.333
497.778	512.000	568.889	625.778	640.000
622.222	622.222	711.111	800.000	800.000"""
order = s.split('\n')
dat = np.array([list(map(float, o.split())) for o in order])
plt.figure(figsize=(11, 5))
plt.subplot(1, 3, 1)
echellogramplot(zip(range(7, 3, -1), dat),
                '$R_{400}=25\ N=2.1$', ylbl=True)

# 17-33
s = """388.571	394.280	400.346	406.412	412.121
400.714	406.406	412.857	419.308	425.000
413.641	419.301	426.175	433.049	438.710
427.429	433.041	440.381	447.721	453.333
442.167	447.712	455.567	463.421	468.966
457.959	463.411	471.837	480.262	485.714
474.921	480.251	489.312	498.374	503.704
493.187	498.360	508.132	517.904	523.077
512.914	517.888	528.457	539.026	544.000
534.286	539.008	550.476	561.944	566.667
557.516	561.923	574.410	586.897	591.304
582.857	586.871	600.519	614.168	618.182
610.612	614.137	629.116	644.095	647.619
641.143	644.057	660.571	677.086	680.000
674.887	677.040	695.338	713.637	715.789
712.381	713.580	733.968	754.356	755.556
754.286	754.286	777.143	800.000	800.000"""
order = s.split('\n')
dat = np.array([list(map(float, o.split())) for o in order])
plt.subplot(1, 3, 2)
echellogramplot(zip(range(33, 16, -1), dat),
                '$R_{400}=50\ N=1.33$')

# 43-21 R100, n2.1
s = """391.111	395.553	400.207	404.860	409.302
400.423	404.858	409.735	414.613	419.048
410.190	414.610	419.729	424.848	429.268
420.444	424.844	430.222	435.600	440.000
431.225	435.596	441.254	446.911	451.282
442.573	446.907	452.865	458.824	463.158
454.535	458.820	465.105	471.390	475.676
467.160	471.385	478.025	484.664	488.889
480.508	484.659	491.683	498.707	502.857
494.641	498.700	506.144	513.587	517.647
509.630	513.580	521.481	529.383	533.333
525.556	529.375	537.778	546.181	550.000
542.509	546.172	555.125	564.079	567.742
560.593	564.069	573.630	583.190	586.667
579.923	583.179	593.410	603.641	606.897
600.635	603.628	614.603	625.578	628.571
622.881	625.563	637.366	649.169	651.852
646.838	649.152	661.880	674.609	676.923
672.711	674.588	688.356	702.123	704.000
700.741	702.099	717.037	731.975	733.333
731.208	731.947	748.213	764.478	765.217
764.444	764.444	782.222	800.000	800.000"""
order = s.split('\n')
dat = np.array([list(map(float, o.split())) for o in order])
plt.subplot(1, 3, 3)
echellogramplot(zip(range(43, 20, -1), dat),
                '$R_{400}=100\ N=2.1$')
plt.subplots_adjust(.05, .125, .98, .88, .11)
