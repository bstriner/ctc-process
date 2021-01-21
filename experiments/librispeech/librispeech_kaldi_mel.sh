MEL_SIZE="${1}"
shift
DATADIR="${1}"
shift

rm -Rf ${DATADIR}

export train_cmd="run.pl --mem 2G"
export decode_cmd="run.pl --mem 4G"
export mkgraph_cmd="run.pl --mem 8G"
export KALDI_ROOT=/opt/kaldi
export S5=/opt/kaldi/egs/librispeech/s5
export PATH=$S5/utils/:$KALDI_ROOT/tools/openfst/bin:$S5:$PATH
source $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
PYTHON='python2.7'

cd /opt/kaldi/egs/librispeech/s5


SRCDIR=/data/MM1/corpora/librispeech/LibriSpeech

for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360; do
PART2=$(echo $part | sed s/-/_/g)
SRCPARTDIR=${SRCDIR}/${part}
PARTDIR=${DATADIR}/${PART2}
EXPORTDIR=${DATADIR}/export/${PART2}
WAV=${PARTDIR}/wav.scp
MFCC=${PARTDIR}/feats-mfcc.ark
PITCH=${PARTDIR}/feats-pitch.ark
MFCCPITCH=${PARTDIR}/feats-mfcc-pitch.ark
MFCCPITCHSCP=${PARTDIR}/feats-mfcc-pitch.scp
CMVN=${PARTDIR}/cmvn.ark
FEATSCMVN=${PARTDIR}/feats-cmvn.ark
SPK2UTT=${PARTDIR}/spk2utt
UTT2SPK=${PARTDIR}/utt2spk
TEXT=${PARTDIR}/text
/opt/kaldi/egs/librispeech/s5/local/data_prep.sh ${SRCPARTDIR}  ${PARTDIR}
compute-mfcc-feats --sample-frequency=16000 --use-energy=false --num-mel-bins=${MEL_SIZE} --num-ceps=${MEL_SIZE} --low-freq=20 --high-freq=-400 scp:${WAV} ark:${MFCC}
compute-kaldi-pitch-feats --sample-frequency=16000 scp:${WAV} ark:- |  process-kaldi-pitch-feats --add-raw-log-pitch=true ark:- ark:${PITCH}
paste-feats --length-tolerance=2 \
  "ark:${MFCC}" \
  "ark:${PITCH}" \
  ark:- | \
  copy-feats ark:- \
  ark:${MFCCPITCH}
compute-cmvn-stats --spk2utt=ark:${SPK2UTT} ark:${MFCCPITCH} ark:${CMVN}
apply-cmvn --utt2spk=ark:${UTT2SPK} ark:${CMVN} ark:${MFCCPITCH} ark:${FEATSCMVN}
mkdir -p ${EXPORTDIR}
cp ${FEATSCMVN} ${EXPORTDIR}
cp ${TEXT} ${EXPORTDIR}
cp ${UTT2SPK} ${EXPORTDIR}
done
