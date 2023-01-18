from django import forms

CHANNEL_CHOICES = [('rgb', 'RGB'), ('yuv', 'YUV')]
DENOISE_CHOICES = [('dwt', 'DWT'), ('tv', 'Total Variation'), ('lsvd', 'Local SVD'), ('none', 'None')]
FEATURE_CHOICES = [('dct', 'DCT'), ('avg', 'Average'), ('abs', 'Absolute avg')]
BSIZE_CHOICES = [('2', 2), ('4', 4), ('8', 8), ('16', 16), ('32', 32)]

class UploadForm(forms.Form):
    img_input = forms.ImageField(required=False)


class CFAAnalysisForm(forms.Form):

    img_path = forms.CharField(widget=forms.HiddenInput, required=False)

    channel = forms.ChoiceField(choices=CHANNEL_CHOICES,)

    denoise = forms.ChoiceField(choices=DENOISE_CHOICES,)

    feature = forms.ChoiceField(choices=FEATURE_CHOICES,)

    bsize = forms.ChoiceField(choices=BSIZE_CHOICES, label='Block size')
    #segment = forms.ChoiceField(choices=SEGMENT_CHOICES,)

    pmap = forms.BooleanField(label="Prob. map", required=False, initial=True)