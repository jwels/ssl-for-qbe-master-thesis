from django.core.management.base import BaseCommand, CommandError
from train.models import TrainedModel
from predict.predict_scripts.predict_gdbss import predict_gdbss
from predict.predict_scripts.predict_dt import predict_dt
from predict.predict_scripts.predict_gdbs import predict_gdbs 
from predict.predict_scripts.predict_dtss import predict_dtss
from predict.predict_scripts.predict_rf import predict_rf
from predict.predict_scripts.predict_rfss import predict_rfss
import pickle, sys

class Command(BaseCommand):
    help = 'Start a training process for the passed model'

    def add_arguments(self, parser):
        parser.add_argument('model_ids', nargs='+', type=int)        

    def startGDBSS(self, model):
        result = predict_gdbss(model.modelParams, model.removeLowestCertaintyPercentage)
        return result

    def startGDBS(self, model):
        result = predict_gdbs(model.modelParams)
        return result

    def startDT(self, model):
        result = predict_dt(model.modelParams)
        return result

    def startDTSS(self, model):
            result = predict_dtss(model.modelParams, model.removeLowestCertaintyPercentage)
            return result

    def startRF(self, model):
        result = predict_rf(model.modelParams)
        return result

    def startRFSS(self, model):
            result = predict_rfss(model.modelParams, model.removeLowestCertaintyPercentage)
            return result


    def handle(self, *args, **options):
        for model_id in options['model_ids']:
            try:
                model = TrainedModel.objects.get(pk=model_id)
            except TrainedModel.DoesNotExist:
                raise CommandError('Model "%s" does not exist' % model_id)
            
            # map the model types to the corresponding predict functions
            modelOptions = {
                "gdbss": self.startGDBSS,
                "gdbs": self.startGDBS,
                "dt": self.startDT,
                "dtss": self.startDTSS,
                "rf": self.startRF,
                "rfss": self.startRFSS,
            }
            
            result = modelOptions[model.modelType](model)
            # print(result)
            # model.f1scoreTest = result[0]
            # model.f1scoreVal = result[1]
            # model.modelParams = result[2]
            # model.finishedTraining = True
            # model.save()

            self.stdout.write(result)
            # self.stdout.write(self.style.SUCCESS('Successfully created prediction using model with ID "%s"' % model_id))