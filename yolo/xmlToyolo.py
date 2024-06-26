from pylabel import importer
dataset = importer.ImportVOC(path='/home/sanju/Downloads/India/train/annotations')
dataset.export.ExportToYoloV5()