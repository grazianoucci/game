
import subprocess
import os

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

if __name__ == "__main__":

    stat = simple_benchmark()

    if(stat == 0):
      print 'everthing seems fine'


