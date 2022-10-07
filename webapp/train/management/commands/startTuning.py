from django.core.management.base import BaseCommand, CommandError
from train.models import TrainedModel
from train.training_scripts.tune_gdbss import hyperopt_gdbss
from train.training_scripts.tune_gdbs import hyperopt_gdbs
from train.training_scripts.tune_dt import hyperopt_dt
from train.training_scripts.tune_dtss import hyperopt_dtss
from train.training_scripts.tune_rf import hyperopt_rf
from train.training_scripts.tune_rfss import hyperopt_rfss
import json

class Command(BaseCommand):
    help = 'Start a training process for the passed model'

    def add_arguments(self, parser):
        parser.add_argument('model_ids', nargs='+', type=int)

    def startRF(self, model):
        result = hyperopt_rf(model)
        print(result)
        return result

    def startRFSS(self, model):
        result = hyperopt_rfss(model)
        print(result)
        return result

    def startGDBSS(self, model):
        # result = tune_gdbss()
        result = hyperopt_gdbss(model)
        print(result)
        return result

    def startGDBS(self, model):
        result = hyperopt_gdbs(model)
        print(result)
        return result

    def startDT(self, model):
        result = hyperopt_dt(model)
        print(result)
        return result

    def startDTSS(self, model):
        result = hyperopt_dtss(model)
        print(result)
        return result

    def handle(self, *args, **options):
        for model_id in options['model_ids']:
            try:
                model = TrainedModel.objects.get(pk=model_id)
            except TrainedModel.DoesNotExist:
                raise CommandError('Model "%s" does not exist' % model_id)

            # map the model types to the corresponding training functions
            modelOptions = {
                "gdbss": self.startGDBSS,
                "gdbs": self.startGDBS,
                "dt": self.startDT,
                "dtss": self.startDTSS,
                "rf": self.startRF,
                "rfss": self.startRFSS
            }

            result = modelOptions[model.modelType](model)
            model.f1scoreTest = result[0]
            model.f1scoreVal = result[1]
            model.modelParams = json.loads(str(result[2]).replace("\'", "\""))
            model.finishedTraining = True
            model.save()

            self.stdout.write(self.style.SUCCESS('Successfully trained model with ID "%s"' % model_id))