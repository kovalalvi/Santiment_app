from django.shortcuts import render
from .models import SentimentModel
from .forms import SentimentForm
from pytorch_model.predict import SentimentAnalyzer

# Create your views here.
def SentimentApp(request):
    form = SentimentForm(request.POST or None)
    context = {}
    if request.method == 'POST':
        if form.is_valid():
            sent = form.cleaned_data.get('Sentence')    # got the sentence
            textAns = SentimentAnalyzer(sent)
            context['prob'] = textAns

            context['prob_round'] = round(context['prob'], 3)
            context['rate'] = int(round(context['prob'], 1)*10)
        else:
            form = SentimentForm()

    context['form'] = form


    return render(request, 'app.html', context=context)
