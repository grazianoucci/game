
import subprocess
import os
import numpy as np

def load_ls(cartella='.'):

    if cartella == '':
        cartella = '.'
    p = subprocess.Popen(['ls',cartella], stdout=subprocess.PIPE)
    out, err = p.communicate()
    nomifile=out.split('\n')
    return nomifile

def retain_format(lista_in,formato='.dat'):

    lista_out = []
    for nome in lista_in:
        end_f=nome[len(nome)-4:len(nome)]
        if(end_f ==formato):
          lista_out.append(nome)
    return lista_out

def simple_benchmark(
     folder_current  = 'output/'
    ,folder_original = 'benchmark/'
    ,megaverbose     = False
    ):

    stat = 0

    original_outs = retain_format(load_ls(folder_original))
    assert len(original_outs) == 32

    for file_name in original_outs[:]:

      old = folder_original + os.path.basename(file_name)
      new = folder_current  + os.path.basename(file_name)
      p = subprocess.Popen(['diff',old,new], stdout=subprocess.PIPE)
      out, err = p.communicate()
      if(out != ''):
        stat = stat+1
        print 'File', file_name
        out_proc = out.split('\n')
        out_proc = out_proc[:-1]
        n_lines = 0
        for line in out_proc:
            if line[0] =='<':
                n_lines = n_lines+1
        print '  line(s) changed',n_lines
        if(megaverbose):
            print out
    return stat

def check_precision(
     folder_current  = 'output/'
    ,folder_original = 'benchmark/'
    ,f_in            = "output_ml.dat"
    ,toll            = 1.e-2
    ,log_data        = True
    ,verbose         = True
    ):

    out = 0

    if(verbose):
        print 'Cheking'
        print '  current :',folder_current
        print '  original:',folder_original
        print '  file    :',f_in
        print '  '
        print '  adopted tollerance'
        print '    ',toll*100,'%'

    new_data = np.loadtxt(folder_current+f_in)
    old_data = np.loadtxt(folder_original+f_in)

    assert(np.all(old_data[:,0] ==new_data[:,0]))

    mean     = old_data[:,1::3]
    error    = old_data[:,3::3]
    mean_new = new_data[:,1::3]

    if(log_data):
        test     = np.abs(10**mean - 10**mean_new)/error
    else:
        test     = np.abs(mean - mean_new)/error
    test     = test>toll

    out      = float(np.sum(test))/np.prod(np.shape(test))

    if(verbose):
      print '  '
      if(out == 0):
          print '  everything seems fine'
      else:
          print '  error in ',out*100,'% of the cases'

    return out

if __name__ == "__main__":

    if(False):
        print 'Diff check'
        stat = simple_benchmark()

        if(stat == 0):
            print 'everything seems fine'
    if(True):
        print 'Output precision check'
        stat = check_precision()
        if(stat == 0):
            print 'everything seems fine'

        stat = check_precision(f_in = "output_ml_additional.dat",log_data=False)

        if(stat == 0):
            print 'everthing seems fine'
