import util

in_path = 'input/highway.mp4'
out_path = 'output/highway'
cleanBG = util.createCleanBG(input_path=in_path, memorize=15, skipping=6, save_result=True, output_path=out_path)
util.segment(cleanBG, input_path=in_path, output_path=out_path)
