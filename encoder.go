package tweetenc

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

// initStddevBias influences the initial standard
// deviations predicted by an Encoder.
// This value is in the log domain.
//
// A negative value means that the predictions will be
// subject to less gaussian noise than they would with
// a standard deviation of 1.
//
// A value of -2 starts the predictions' stddev values at
// roughly e^-2.
const initStddevBias = -2

func init() {
	var e Encoder
	serializer.RegisterTypedDeserializer(e.SerializerType(), DeserializeEncoder)
}

// An Encoder encodes strings of bytes into vectors.
type Encoder struct {
	Block         anyrnn.Block
	MeanEncoder   anynet.Layer
	StddevEncoder anynet.Layer
}

// DeserializeEncoder deserializes an Encoder.
func DeserializeEncoder(d []byte) (*Encoder, error) {
	var block anyrnn.Block
	var mean, stddev anynet.Layer
	if err := serializer.DeserializeAny(d, &block, &mean, &stddev); err != nil {
		return nil, errors.New("deserialize Encoder: " + err.Error())
	}
	return &Encoder{
		Block:         block,
		MeanEncoder:   mean,
		StddevEncoder: stddev,
	}, nil
}

// NewEncoder creates an Encoder.
func NewEncoder(c anyvec.Creator, encodedSize, stateSize int) *Encoder {
	scaler := c.MakeNumeric(16)
	stddevLayer := anynet.NewFC(c, stateSize, encodedSize)
	stddevLayer.Biases.Vector.AddScalar(c.MakeNumeric(initStddevBias))
	return &Encoder{
		Block: anyrnn.Stack{
			anyrnn.NewLSTM(c, 0x100, stateSize).ScaleInWeights(scaler),
			anyrnn.NewLSTM(c, stateSize, stateSize).ScaleInWeights(scaler),
			anyrnn.NewLSTM(c, stateSize, stateSize).ScaleInWeights(scaler),
		},
		MeanEncoder: anynet.Net{
			anynet.NewFC(c, stateSize, encodedSize),
		},
		StddevEncoder: anynet.Net{
			stddevLayer,
		},
	}
}

// Apply applies the encoder to an input sequence, which
// should be reversed and should lack a null-terminator.
//
// There must be at least one sequence, and all sequences
// must be non-empty.
//
// The resulting mean and log-scale standard deviations
// are returned, in that order.
func (e *Encoder) Apply(s anyseq.Seq) anydiff.MultiRes {
	if len(s.Output()) == 0 {
		panic("must have at least one sequence")
	}
	if s.Output()[0].NumPresent() != len(s.Output()[0].Present) {
		panic("input sequences must be non-empty")
	}
	outSeq := anyrnn.Map(s, e.Block)
	tail := anyseq.Tail(outSeq)

	temp := anydiff.Fuse(tail)
	return anydiff.PoolMulti(temp, func(reses []anydiff.Res) anydiff.MultiRes {
		tail = reses[0]
		n := s.Output()[0].NumPresent()
		means := e.MeanEncoder.Apply(tail, n)
		stddevs := e.StddevEncoder.Apply(tail, n)
		return anydiff.Fuse(means, stddevs)
	})
}

// Encode encodes strings to a packed vector of the most
// probable encodings.
func (e *Encoder) Encode(samples ...string) (mean, logStddev anyvec.Vector) {
	var inSeqs [][]anyvec.Vector
	cr := e.Block.(anynet.Parameterizer).Parameters()[0].Vector.Creator()
	for _, s := range samples {
		inSeq := []anyvec.Vector{}
		byteString := []byte(s)
		for i := len(byteString) - 1; i >= 0; i-- {
			inSeq = append(inSeq, oneHot(cr, byteString[i]))
		}
		inSeqs = append(inSeqs, inSeq)
	}
	seqs := anyseq.ConstSeqList(cr, inSeqs)
	out := e.Apply(seqs).Outputs()
	return out[0], out[1]
}

// SerializerType returns the unique ID used to serialize
// an Encoder with the serializer package.
func (e *Encoder) SerializerType() string {
	return "github.com/unixpickle/tweetenc.Encoder"
}

// Serialize serializes the Encoder.
func (e *Encoder) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		e.Block,
		e.MeanEncoder,
		e.StddevEncoder,
	)
}
