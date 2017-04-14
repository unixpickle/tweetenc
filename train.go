package tweetenc

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Trainer is an anysgd.Fetcher and anysgd.Gradienter
// for training an encoder/decoder pair.
type Trainer struct {
	Encoder *Encoder
	Decoder *Decoder

	// KL determines how much effect the KL-divergence term
	// has on the cost.
	// As this increases towards 1, the auto-encoder becomes
	// more and more of a VAE.
	KL float64

	// LastCost is set every time Gradient is called.
	LastCost anyvec.Numeric

	// Iteration is incremented for every Gradient call and
	// is passed to KLAmount to compute the rate.
	Iteration int
}

// Fetch produces a batch that represents the training
// samples in the SampleList.
func (t *Trainer) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	cr := t.creator()
	zero := oneHot(cr, 0)

	inSeqs := make([][]anyvec.Vector, s.Len())
	guideSeqs := make([][]anyvec.Vector, s.Len())
	for i := range inSeqs {
		data := s.(SampleList)[i]
		if len(data) == 0 {
			return nil, errors.New("encountered empty sample string")
		}
		seq := []anyvec.Vector{zero}
		for _, x := range data {
			seq = append(seq, oneHot(cr, x))
		}
		seq = append(seq, zero)
		inSeqs[i] = seq[1:]
		guideSeqs[i] = seq[:len(seq)-1]
	}

	revIn := make([][]anyvec.Vector, s.Len())
	for i, seq := range guideSeqs {
		var rev []anyvec.Vector
		// Skip index 0 since that's a null-terminator.
		for j := len(seq) - 1; j > 0; j-- {
			rev = append(rev, seq[j])
		}
		revIn[i] = rev
	}

	return &trainerBatch{
		ReversedIn: anyseq.ConstSeqList(cr, revIn),
		Desired:    anyseq.ConstSeqList(cr, inSeqs),
		Guide:      anyseq.ConstSeqList(cr, guideSeqs),
	}, nil
}

// TotalCost computes the average cost for the batch.
func (t *Trainer) TotalCost(b anysgd.Batch) anydiff.Res {
	tb := b.(*trainerBatch)
	batchSize := len(tb.Desired.Output()[0].Present)
	multiEnc := t.Encoder.Apply(tb.ReversedIn)
	res := anydiff.PoolMulti(multiEnc, func(reses []anydiff.Res) anydiff.MultiRes {
		mean := reses[0]
		logStddev := reses[1]

		c := mean.Output().Creator()

		stddev := anydiff.Exp(logStddev)
		noise := c.MakeVector(mean.Output().Len())
		anyvec.Rand(noise, anyvec.Normal, nil)
		sampled := anydiff.Add(mean, anydiff.Mul(anydiff.NewConst(noise), stddev))

		decoded := t.Decoder.Guided(sampled, tb.Guide, batchSize)

		var idx int
		var costCount int
		allCosts := anyseq.Map(decoded, func(a anydiff.Res, n int) anydiff.Res {
			desired := tb.Desired.Output()[idx]
			costCount += desired.NumPresent()
			idx++
			c := anynet.DotCost{}.Cost(anydiff.NewConst(desired.Packed), a, n)
			return c
		})

		sum := anydiff.Sum(anyseq.Sum(allCosts))

		variances := anydiff.Square(stddev)
		klDivergence := anydiff.AddScalar(
			anydiff.Add(anydiff.Sum(variances), anydiff.Dot(mean, mean)),
			c.MakeNumeric(float64(-mean.Output().Len())),
		)
		klDivergence = anydiff.Scale(klDivergence, c.MakeNumeric(0.5))
		klDivergence = anydiff.Sub(klDivergence, anydiff.Sum(logStddev))
		klDivergence = anydiff.Scale(klDivergence, c.MakeNumeric(t.KL))

		scaler := c.MakeNumeric(1 / float64(costCount))
		return anydiff.Fuse(anydiff.Scale(anydiff.Add(sum, klDivergence), scaler))
	})
	return anydiff.Unfuse(res, func(reses []anydiff.Res) anydiff.Res {
		return reses[0]
	})
}

// Gradient computes a gradient for the batch and also
// sets t.LastCost.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	res := anydiff.Grad{}
	parts := []interface{}{t.Encoder.Block, t.Decoder.Block, t.Decoder.StateMapper}
	for _, part := range parts {
		if parameterizer, ok := part.(anynet.Parameterizer); ok {
			for _, p := range parameterizer.Parameters() {
				res[p] = p.Vector.Creator().MakeVector(p.Vector.Len())
			}
		}
	}
	cost := t.TotalCost(b)
	t.LastCost = anyvec.Sum(cost.Output())
	data := cost.Output().Creator().MakeNumericList([]float64{1})
	upstream := cost.Output().Creator().MakeVectorData(data)
	cost.Propagate(upstream, res)
	return res
}

func (t *Trainer) creator() anyvec.Creator {
	return t.Decoder.Block.Parameters()[0].Vector.Creator()
}

type trainerBatch struct {
	ReversedIn anyseq.Seq
	Desired    anyseq.Seq
	Guide      anyseq.Seq
}

func oneHot(c anyvec.Creator, b byte) anyvec.Vector {
	data := make([]float64, 0x100)
	data[int(b)] = 1
	return c.MakeVectorData(c.MakeNumericList(data))
}
