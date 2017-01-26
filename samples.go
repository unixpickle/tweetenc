package tweetenc

import (
	"crypto/md5"
	"encoding/csv"
	"errors"
	"os"

	"github.com/unixpickle/anynet/anysgd"
)

// A SampleList is a list of textual samples.
type SampleList [][]byte

// ReadSampleList reads a CSV file full of text samples.
//
// The last column is used as the text body.
// The other columns are ignored.
func ReadSampleList(csvPath string) (SampleList, error) {
	f, err := os.Open(csvPath)
	if err != nil {
		return nil, errors.New("read samples: " + err.Error())
	}
	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, errors.New("read samples: " + err.Error())
	}
	var samples [][]byte
	for _, record := range records {
		if len(record) == 0 {
			return nil, errors.New("read samples: empty row")
		}
		sample := []byte(record[len(record)-1])
		if len(sample) == 0 {
			continue
		}
		samples = append(samples, sample)
	}
	return samples, nil
}

// Len returns the sample count.
func (s SampleList) Len() int {
	return len(s)
}

// Swap swaps two samples.
func (s SampleList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Slice returns a shallow copy of the slice.
func (s SampleList) Slice(start, end int) anysgd.SampleList {
	return append(SampleList{}, s[start:end]...)
}

// Hash returns a hash of the sample.
func (s SampleList) Hash(i int) []byte {
	res := md5.Sum(s[i])
	return res[:]
}
