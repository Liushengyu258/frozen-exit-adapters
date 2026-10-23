#!/usr/bin/env python3
"""Restore and verify the full experiment snapshot from its GitHub Release."""
import argparse,gzip,hashlib,json,os,shutil,tarfile,tempfile,urllib.request
from pathlib import Path,PurePosixPath
BASE='https://github.com/Liushengyu258/frozen-exit-adapters/releases/download/server-data-20261023/'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
 dest=root.joinpath(*p.parts)
 for ancestor in [dest,*dest.parents]:
  if ancestor==root.parent:break
  if ancestor.is_symlink():raise ValueError('Existing symbolic link blocks restoration: '+str(ancestor))
 return dest
def fetch(url,path,checksum=None):
 with urllib.request.urlopen(url,timeout=180) as src,path.open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
 if checksum and sha(path)!=checksum:raise ValueError('Downloaded asset SHA256 mismatch: '+url)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',required=True);a=p.parse_args()
 root=Path(a.destination).absolute()
 if root.exists() and any(root.iterdir()):raise ValueError('Destination must be empty to prevent overwriting files')
 root.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='restore-',dir=root.parent) as tmp:
  tmp=Path(tmp);manifest=tmp/'manifest.json.gz';fetch(BASE+'manifest.json.gz',manifest)
  with gzip.open(manifest,'rt') as f:m=json.load(f)
  if not m.get('complete'):raise ValueError('Snapshot is incomplete')
  for e in m['entries']:
   if e['type']=='directory':safe(root,e['path']).mkdir(parents=True,exist_ok=True)
  archive=tmp/'small.tar.gz';row=m['small_archive'];fetch(row['url'],archive,row['sha256'])
  with tarfile.open(archive,'r:gz') as tar:
   for member in tar:
    if not member.isfile():raise ValueError('Unexpected non-regular tar member')
    dest=safe(root,member.name);dest.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,dest.open('wb') as out:shutil.copyfileobj(src,out,8*1024**2)
  archive.unlink()
  for name,info in m['large_files'].items():
   print('Restoring',name,flush=True);dest=safe(root,name);dest.parent.mkdir(parents=True,exist_ok=True)
   with dest.open('wb') as out:
    for _,part in sorted(info['parts'].items(),key=lambda kv:int(kv[0])):
     archive=tmp/'part.gz';fetch(part['url'],archive,part['sha256']);h=hashlib.sha256();n=0
     if out.tell()!=part['offset']:raise ValueError('Invalid part offset')
     with gzip.open(archive,'rb') as src:
      for b in iter(lambda:src.read(8*1024**2),b''):h.update(b);n+=len(b);out.write(b)
     if n!=part['raw_size'] or h.hexdigest()!=part['raw_sha256']:raise ValueError('Invalid uncompressed part')
     archive.unlink()
   if sha(dest)!=info['sha256']:raise ValueError('Full file checksum mismatch: '+name)
  for e in m['entries']:
   if e['type']=='file':
    dest=safe(root,e['path'])
    if dest.stat().st_size!=e['size'] or sha(dest)!=e['sha256']:raise ValueError('File verification failed: '+e['path'])
    os.chmod(dest,e['mode']);os.utime(dest,ns=(e['mtime_ns'],e['mtime_ns']))
  for e in m['entries']:
   if e['type']=='symlink':
    dest=safe(root,e['path']);dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(e['target'])
  for e in reversed(m['entries']):
   if e['type']=='directory':
    dest=root/e['path'];os.chmod(dest,e['mode']);os.utime(dest,ns=(e['mtime_ns'],e['mtime_ns']))
 print('Verified and restored',sum(e['type']=='file' for e in m['entries']),'files to',root)
if __name__=='__main__':main()
