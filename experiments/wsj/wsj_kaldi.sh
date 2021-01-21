source cmd.sh
source path.sh

x=test_eval92
apply-cmvn --norm-means --norm-vars \
    --utt2spk=ark:data/$x/utt2spk \
    scp:data/$x/cmvn.scp scp:data/$x/feats.scp \
    ark,t:textdata/$x/feats-normalized.ark

for x in test_eval92 test_eval93 test_dev93 train_si284; do
 steps/make_mfcc_pitch.sh \
--mfcc-config conf/mfcc_hires.conf \
--pitch-config conf/pitch.conf \
--pitch_postprocess_config conf/pitch_process.conf \
--paste_length_tolerance 30 \
--cmd "$train_cmd" --nj 20 \
data/$x data/$x/mfcc_pitch_log data/$x/mfcc_pitch_data || exit 1;
steps/compute_cmvn_stats.sh \
data/$x data/$x/cmvn_log data/$x/cmvn_data|| exit 1;
mkdir -p textdata/$x|| exit 1;
copy-matrix scp:data/$x/feats.scp ark,t:textdata/$x/text-feats.ark || exit 1;
copy-matrix scp:data/$x/cmvn.scp ark,t:textdata/$x/text-cmvn.ark || exit 1;
copy-matrix scp:data/$x/wav.scp ark,t:textdata/$x/text-wav.ark || exit 1;
apply-cmvn --norm-means=true --norm-vars=true \
    --utt2spk=ark:data/$x/utt2spk \
    scp:data/$x/cmvn.scp scp:data/$x/feats.scp \
    ark,t:textdata/$x/feats-normalized.ark || exit 1;
cp data/$x/utt2num_frames textdata/$x || exit 1;
cp data/$x/utt2spk textdata/$x || exit 1;
cp data/$x/spk2gender textdata/$x || exit 1;
cp data/$x/text textdata/$x || exit 1;
done
