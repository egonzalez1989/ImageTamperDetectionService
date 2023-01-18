
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.core.files.storage import default_storage
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django import template
from .forms import *
import os
from django.core.files.base import ContentFile
from .image_analysis.ImageProcessingTools import *
import hashlib
import json

def DCTTest(data, B = 8):
    h, w = data.shape
    hh, ww = h//B, w//B
    F = np.array(block_map(data, lambda X: cv2.dct(X)[-1, -1], B))  # Extract a vectors of features for each block
    F = cv2.resize(F.reshape(hh, ww), None, fx=B, fy=B)
    return F

def avgTest(data, B=8):
    h, w = data.shape
    hh, ww = h // B, w // B
    print(hh, ww)
    avg = block_mean(data, B)
    F = np.array(block_map(data, lambda X: block_corr_similarity(X, avg), B))  # Extract a vectors of features for each block)
    F = cv2.resize(F.reshape(hh, ww), None, fx=B, fy=B)
    return F

#def absTest(data, B=8):
#    return avgTest(np.abs(data), B)


denoise_choices = {'none': lambda I: I, 'dwt': dwt_denoise, 'tv': tv_denoise, 'lsvd': slideSVDPredict}
channel_choices = {'rgb': cv2.COLOR_BGR2RGB, 'yuv': cv2.COLOR_BGR2YUV}
#cmap_choices = {'r': 'Reds', 'g': 'Greens', 'b': 'Blues', 'y': 'gray', 'u': 'YlOrBr', 'v': 'YlGnBu'}
cmap_choices = {'r': 'gray', 'g': 'gray', 'b': 'gray', 'y': 'gray', 'u': 'gray', 'v': 'gray'}
feature_choices = {'dct': DCTTest, 'avg': avgTest, 'abs': avgTest}
segment_choices = {'none': lambda X: X, 'otsu': otsu_components}

#@login_required(login_url="/login/")
def index(request):
    context = {}
    context['upload_form'] = UploadForm()
    return render(request, "index.html", context)

#@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        
        load_template = request.path.split('/')[-1]
        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))
        
    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'error-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:
    
        html_template = loader.get_template( 'error-500.html' )
        return HttpResponse(html_template.render(context, request))


# wpath: Working path create for an image
# cpath: Channel paths
# npath: Noise paths
# dctpath: DCT analysis path

def analysis(request):
    if request.method == 'POST':
        form = CFAAnalysisForm(request.POST)
        if form.is_valid():
            cfa_analysis(request, form)
            return render(request, "analysis.html", {"success": True, "cfa_analysis_form": form, "upload_form": UploadForm()})
        else:
            return render(request, "analysis.html", {"success": True, "cfa_analysis_form": form, "upload_form": UploadForm()})

def upload(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            handle_request(request)
            return render(request, "analysis.html", {"success": True, "cfa_analysis_form": CFAAnalysisForm(), "upload_form": UploadForm()})
    return render(request, "index.html", {"success": False, "upload_form": UploadForm()})


def cfa_analysis(request, form):
    imgdir = 'core/' + request.session['imgdir']
    imgfile = imgdir + '/img.png'
    img = cv2.imread(imgfile)
    print(imgfile)

    # Getting form data
    cvt = form.cleaned_data['channel']
    dnoise = form.cleaned_data['denoise']
    pmap = form.cleaned_data['pmap']
    feat = form.cleaned_data['feature']
    bsize = int(form.cleaned_data['bsize'])
    #segment = form.cleaned_data['segment']

    # Handle channel options
    wdir = imgdir
    cfiles = ['{}/{}'.format(wdir, c) for c in cvt]
    if not os.path.isfile(cfiles[0] + '.npy'):    # Verify if it has been processed before
        if cvt != 'rgb': img = cv2.cvtColor(img, channel_choices[cvt])             # Transform colorspace
        channels = cv2.split(img)
        [np.save(X[0], X[1]) for X in zip(cfiles, channels)]
        [plt.imsave(X[0] + '.png', X[1], cmap=cmap_choices[X[2]]) for X in zip(cfiles, channels, cvt)]

    # Handle denoise options
    wdir = wdir + '/' + dnoise
    createdir(wdir)
    dfiles = ['{}/{}'.format(wdir, c) for c in cvt]
    if not os.path.isfile(dfiles[0]+ '.npy'):
        imgs = [np.load(f + '.npy') for f in cfiles]
        noises = [denoise_choices[dnoise](I) - I for I in imgs]
        [np.save(X[0], X[1]) for X in zip(dfiles, noises)]
        [plt.imsave(X[0]+ '.png', X[1], cmap=cmap_choices[X[2]]) for X in zip(dfiles, noises, cvt)]

    # If a prob. map is required
    if feat == 'abs':
        wdir = wdir + '/abs'
        print(wdir)
        createdir(wdir)
        pfiles = ['{}/{}'.format(wdir, c) for c in cvt]
        if not os.path.isfile(pfiles[0] + '.npy'):
            noises = [np.abs(np.load(f + '.npy')) for f in dfiles]
            [np.save(X[0], X[1]) for X in zip(pfiles, noises)]
            [plt.imsave(X[0] + '.png', X[1], cmap=cmap_choices[X[2]]) for X in zip(pfiles, noises, cvt)]

    # If a prob. map is required
    if pmap:
        wdir = wdir + '/pmap'
        print(wdir)
        createdir(wdir)
        pfiles = ['{}/{}'.format(wdir, c) for c in cvt]
        if not os.path.isfile(pfiles[0] + '.npy'):
            noises = [np.load(f + '.npy') for f in dfiles]
            pmaps = [pmap_erfc(N) for N in noises]
            [np.save(X[0], X[1]) for X in zip(pfiles, pmaps)]
            [plt.imsave(X[0]+ '.png', X[1], cmap=cmap_choices[X[2]]) for X in zip(pfiles, pmaps, cvt)]

    # extract features
    wdir = wdir + '/' + feat
    createdir(wdir)
    wdir = wdir + '/' + str(bsize)
    createdir(wdir)
    ffiles = ['{}/{}'.format(wdir, c) for c in cvt]
    if not os.path.isfile(ffiles[0] + '.npy'):
        if pmap: data = [np.load(f + '.npy') for f in pfiles]
        else: data = [np.load(f + '.npy') for f in dfiles]
        fmaps = [feature_choices[feat](X, B=bsize) for X in data]
        [np.save(X[0], X[1]) for X in zip(ffiles, fmaps)]
        [plt.imsave(X[0] + '.png', X[1], cmap=cmap_choices[X[2]]) for X in zip(ffiles, fmaps, cvt)]
    else:
        fmaps = [np.load(X) for X in ['{}/{}.npy'.format(wdir, c) for c in cvt]]

    hist1 = np.histogram(fmaps[0], bins=100, density=True)
    hist2 = np.histogram(fmaps[1], bins=100, density=True)
    hist3 = np.histogram(fmaps[2], bins=100, density=True)
    request.session['hist1'] = json.dumps({'X': hist1[0].tolist(), 'Y': hist1[1].tolist()})
    request.session['hist2'] = json.dumps({'X': hist2[0].tolist(), 'Y': hist2[1].tolist()})
    request.session['hist3'] = json.dumps({'X': hist3[0].tolist(), 'Y': hist3[1].tolist()})

    request.session['channel1'] = ffiles[0][5:] + '.png'
    request.session['channel2'] = ffiles[1][5:] + '.png'
    request.session['channel3'] = ffiles[2][5:] + '.png'

def handle_request(request):
    f = request.FILES['img_input']
    tmpfile = 'core/static/analysis/{}'.format(f.name)
    default_storage.save(tmpfile, ContentFile(f.read()))
    dirname = md5('core/static/analysis/{}'.format(f.name))

    # Creating working paths and storing image
    imgdir = 'core/static/analysis/' + dirname
    createdir(imgdir)
    img = cv2.imread(tmpfile)
    print(img.shape)
    img = img[:img.shape[0]-(img.shape[0] % 32), :img.shape[1]-(img.shape[1] % 32), :]
    print(img.shape)
    cv2.imwrite('core/static/analysis/' + dirname + '/img.png', img)
    os.remove(tmpfile)

    # session variables to store image directory and location
    request.session['imgdir'] = imgdir[5:]          # Main working path
    request.session['cname'] = dirname

    # Storing BGR channels
    b, g, r = cv2.split(img)
    [np.save(imgdir + '/{}'.format(X[0]), X[1]) for X in (('r', r), ('g', g), ('b', b))]
    [plt.imsave(imgdir + '/{}.png'.format(X[0]), X[1], cmap='gray') for X in (('r', r), ('g', g), ('b', b))]
    request.session['channel1'] = imgdir[5:] + '/r.png'
    request.session['channel2'] = imgdir[5:] + '/g.png'
    request.session['channel3'] = imgdir[5:] + '/b.png'

    hist1 = np.histogram(r, bins=100, density=True)
    hist2 = np.histogram(g, bins=100, density=True)
    hist3 = np.histogram(b, bins=100, density=True)
    request.session['hist1'] = json.dumps({'X': hist1[0].tolist(), 'Y': hist1[1].tolist()})
    request.session['hist2'] = json.dumps({'X': hist2[0].tolist(), 'Y': hist2[1].tolist()})
    request.session['hist3'] = json.dumps({'X': hist3[0].tolist(), 'Y': hist3[1].tolist()})
    request.session['h'] = img.shape[0]
    request.session['w'] = img.shape[1]






def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def createdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
