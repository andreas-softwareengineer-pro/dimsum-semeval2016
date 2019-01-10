// A machine learing system for Detection of Semantic Units and their Meanings
// Developed as a SemEval 2016 task 10 solution
// Author: Andrey Scherbakov, andreas@softwareengineer.pro
// 
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <set>
#include <map>
#include <stdlib.h>
#include <strings.h>

#define USE_WORDVEC (1<<0)
#define USE_HASH (1<<1)
#define USE_DIST (1<<2)
#define USE_WORDWISE (1<<3)
#define RECURSE_COMPOSE (1<<4)

//Uncomment to debug MultiWord learning loop
//    #define DEBUG_MW_SEL 1

using namespace std;
using namespace cnn;
using namespace cnn::expr;

const char* MW_LOG_ID = "isMW";
//Update when heuristic features are added or deleted
const int nMiscFeatures = 15   ;
//Update when distance features are added or deleted
const int nDistFeatures = 4 ;

// ====== Hardcoded adjustments =======
//Learning rate
const float rate  = 0.15;
//Max number of epochs
const int NEPOCH = 1024;
//Default dimension of n-word incremental number
const int DEFAULT_COMPOSED_SIZE = 128;
//Layer sizes in Multi Word classificator perceptron
//Note: Currently it has 3 layers, still SemEval 10 results are for 2 layer perceptron
const int HIDDEN2_SIZE = 256;
const int HIDDEN3_SIZE = 256;
//Layer size in Supersense classificator perceptron
const int S_HIDDEN2_SIZE = 128;

// Lookup lemma in word2vec dictionary if a word itself not found
#define LEMMA_LOOKUP 1
// One character spell correction option (dont use it unless stop words are present in a dictionary!)
// #define LOOKUP_CORRECTION 1

//Print a cnn expression
ostream& pExp(ostream& o, Expression * x, size_t  limit = 1000000) {
	vector<float>a (as_vector(x->value() ));
	for (size_t i = 0; i < min(a.size(), limit); ++i) o << a[i] << "\t";
	return o;
}

void pExp(Expression * x) {
	vector<float>a (as_vector(x->value() ));
	for (size_t i = 0; i < min(a.size(), 10UL); ++i) cerr << a[i] << " ";
	cerr << endl;
}

ostream& logVectors(ostream& o)
{
		return o;
}

template <typename ... T> ostream& logVectors(ostream& o, Expression* v0, T... varr)
{
		pExp(o<<"\t", v0, 10); 
		logVectors(o,varr...);
		return o;
}


template <typename I, typename ... T> void logVectors(ostream& o, char t, I v, T ... varr) {
	o <<t<<"\t"<<v ;
	logVectors(o, varr...);
	o<<endl;;
}

//default ratio of Valiation set in a train set
static float devRatio = 0.0;

//Correlation data container
class CorrelationTracker {
	public:
	bool accepting;
	private:
	 struct Acc {
		float xsum, xsqsum, ysum, ysqsum, prod;
		int n;
		void add (float x, float y) {xsum+=x, xsqsum+=x*x, ysum+=y, ysqsum+=y*y, ++n, prod+= x*y ; }
		Acc() :xsum(0.), xsqsum(0.), ysum(0.), ysqsum(0.), n(0), prod(0.) {}
		float getCorr() {return (n * prod - xsum*ysum) / sqrt((xsqsum * n - xsum*xsum)*(ysqsum * n - ysum*ysum));}
	};
	
	map<string,map<string,Acc> > nomination;
	public:
	void add(const string&  , bool) {}
	template <typename ... T> void add(const string & id, bool y, Expression x, const char** x_chart, T ... varr)
	{
		if (!accepting) return;
		vector <float> xv (as_vector(x. value()));
		for (size_t i=0; i<xv.size(); ++i) {
			nomination[id][x_chart[i]].add(xv[i],y?1.:0.);
		}
		add(id,y,varr...);
	}
	ostream & operator >> (ostream& os) {
		for (auto& yit: nomination) {
			os << yit.first << ": ";
			for (auto& xit: yit.second) os << " " << xit.first <<": " << xit .second.getCorr() << ";";
			os << endl;
		}
	}
	CorrelationTracker(): accepting(false) {}
};	

// Word lookup debug log file (used in -d mode)
ofstream wlookuplog;

// Feature to MWE/Supersence correlation table output file (used in -d mode)
ofstream correlation_tab;
CorrelationTracker corr;

cnn::real plusminusbit(bool v) {
	return v? 1.:-1.;
}

//Uncomment to download feature vector values at training stage
// #define LOG_FEATURE_VALUES 1
#ifdef LOG_FEATURE_VALUES
ofstream vf("mw.vec");
ofstream vsf("sense.vec");
#endif //LOG_FEATURE_VALUES

//returns a vector of 1. / -1. values representing each bit in a given unsigned integer
template <class I> vector<cnn::real> bitvec(I mi, size_t  sz) {
      vector<cnn::real> r(sz);
      for (unsigned i = 0; i < sz; ++i) {
      		bool x = (((mi >> i) & 1U) != 0);
      		r[i] = x ? 1. : -1.;
      }
      return r;
}

bool  hasNan (const vector<float> &a) {
	for (size_t i = 0; i < a.size(); ++i) if (isnan(a[i])) return true;
	return false;
}

bool  hasNan (Expression * x) {
	vector<float>a (as_vector(x->value() ));
	for (size_t i = 0; i < a.size(); ++i) if (isnan(a[i])) return true;
	return false;
}

//djb2 by Dan Bernstein
uint64_t
hash(const char *str)
{
    uint64_t hash = 5381;
    int c;

    while (c = (unsigned char) *str++)
        if (isalpha(c)) hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

Expression oneHot(ComputationGraph& cg, size_t i, size_t sz) {
	vector<Expression> v(sz,input(cg,kSCALAR_MINUSONE));
	v[i] = input(cg,kSCALAR_ONE);
	return concatenate(v);
}

long int HASH_DIM = 64;

typedef size_t Sense,POS;

//Vocabulary element
struct WordInfo {
	WordInfo(unsigned long long int r, size_t dim):range(r),v(dim,0.){}
	long long int range;
	vector<float> v;
};

WordInfo unknownWord = WordInfo(ULLONG_MAX,0U);

struct Vocab {
	long vsize;
	unordered_map< string, WordInfo > words;
#ifdef LOOKUP_CORRECTION
	//Derived 
	set<char> smallLetters;

	void initDerived() {
		auto it=words.begin();
		size_t i = 0;
		for (; i<1000 && it!=words.end(); ++i,++it) {
			for (size_t j=0; j<it->first. length(); ++j) if (islower(it->first[j])) {
				smallLetters.insert(it->first[j]);
			}
		}
	}
#endif // LOOKUP_CORRECTION
	Vocab() :vsize(0) {}
};

//Word2Vec vocabulary
Vocab voc;

bool hasNumber(const char* s) {
	return strpbrk(s,"0123456789");}
bool isNumber(const char* s) {return hasNumber(s) &&
	!s[strspn(s,"0123456789-., ")];}

inline void skipNonAlphaChar(const char* &word) {while (word[0] && !isalpha(word[0]) ) word++;}
inline bool hasNonAlphaChar (const char* word) {
	while (word[0]) {
		if (!isalpha(word[0])&&!strchr("#@",word[0])) return true;
		word++;
	}
   return false;
}

inline void skipTagChar(const char* &word) {while (word[0] && strchr("#@",word[0]) ) word++;}
inline bool isCapital(const char* word) {
	skipNonAlphaChar(word);
	return isupper(*word);
}
inline float upperRatio(const char* word) {
	skipNonAlphaChar(word);
	size_t u=0, l=0;
	for (;*word;++word) if (isupper(*word)) ++u; else if (islower(*word)) ++l;
	return u? ((float) u)/(u+l): 0.;
}

ofstream& logLookupResult(const string& in, const pair<string, WordInfo> & out)
{
	if (hasNan(out. second.v)) cerr << "NaN word: " << in << endl;
	if (wlookuplog.is_open()) wlookuplog << "lookup\t" << in << "\tas\t" << out.first << endl;
}
ofstream& logLookupUnknown(const string& in)
{
	if (wlookuplog.is_open() ) wlookuplog << "lookup\t" << in << "\tas\tUnknown" << endl;
} 

// Dictionary lookup for a word
inline const WordInfo& lookup(const Vocab& voc, const char* word, const char* lemma) {
	const char* pw = word;
	skipTagChar(pw);
	if (isNumber(pw)) pw = "NUMBER";
	auto i = voc.words.find(pw);
	if (i != voc.words.end())  { logLookupResult(word, *i); return i->second; }
	// lowercase the word
	string low(word) ;
	std::transform(low.begin(), low.end(), low.begin(), ::tolower);
	i = voc.words.find(low);
	if (i != voc.words.end()) {logLookupResult(word, *i); return i->second; }
#ifdef LEMMA_LOOKUP
	i = voc.words.find(lemma);
	if (i != voc.words.end()) {logLookupResult(word, *i); return i->second; }
#endif // LEMMA_LOOKUP
	// remove all non-alpha characters
	char purealpha[low.length()+1];
	* std::copy_if(low.begin(),low.end(),purealpha,
		[](const char c){return isalpha(c);}) = 0;
	i = voc.words.find(purealpha);
	if (i != voc.words.end()) {logLookupResult(word, *i); return i->second; }

#ifdef LOOKUP_CORRECTION
	w = b;
	for (size_t pos=0; pos<w.length();pos++) {
		char old = w[pos];
		for (auto l:voc.smallLetters) if (l!=old) {
			w[pos]=l;
			auto j = voc.words.find(w);
			if (j!=voc.words.end() && (i==voc.words.end() || j->second.range < i->second.range) ) i=j;
			}
		w[pos]=old;
	}
	if (i != voc.words.end()) { logLookupResult(word, *i); return i->second; }
#endif //LOOKUP_CORRECTION
	logLookupUnknown(word);
	return unknownWord;
}

struct UnexpectedEntityException: public exception {
	string msg;
	UnexpectedEntityException(const string &name) : msg(string("New entity encountered while model is closed: ")+name) {}
	virtual const char* what() const noexcept {return msg.c_str();  }
};

void serialize(boost::archive::text_oarchive& oa, const string& s, unsigned int ver) {oa & s;}
void serialize(boost::archive::text_iarchive& ia, string& s, unsigned int ver) {ia >> s;}
	
// A structure to collect and lookup string <-> integer Id relations
class NameHash {
	map<string,int> s2i;
	vector<const string*> i2s;
	bool& lock;

	public:
	int operator [] (const string& s) {
		auto i = s2i.insert(pair<string,int>(s,s2i.size()));
		if (i.second) 
			if (lock) throw new UnexpectedEntityException("New entity encountered while model is closed: "+s);
			else i2s.push_back(&i.first->first);
		assert(i2s.size()==s2i.size());
		return i.first->second;
		}
	NameHash(bool &l) : lock(l) {}
	size_t size() {return s2i.size();}

	void serialize(boost::archive::text_oarchive& oa, unsigned int ver) {
		size_t sz = size();
		oa << sz;
		for (size_t j=0; j<size(); ++j) {string s(*i2s[j]);oa<<s;}
	}

	void serialize(boost::archive::text_iarchive& ia, unsigned int ver) {
		i2s.clear();
		s2i.clear();
		size_t sz;
		ia >> sz;
		for (size_t j=0; j<sz; ++j) {
			string s;
			ia >> s;
			(*this)[s];
		}
	}
	
	const string& at(size_t i) {return *i2s.at(i);}
};

struct MWCompModel {
bool initialized;
struct Tables  { NameHash pos, senses ; } tables;
	unsigned  int distPoints, feature_mask;
	int composedSize;
    LookupParameters *compose_W0, *compose_W1, *compose_Wh;
	Parameters  *W2, * W2d, * W3, * W4, /** Wm,*/ * Wf, * Wd, * Wp, * b1, * b2, * b3,
		* b4, * W2s, * W3s, * b2s, * b3s, *v0;
	Model m;

	bool initialize() {
	  return initialize(this->tables.pos.size(), this->tables.senses.size());
	}
	bool initialize(int posC, int senseC) {
	  Parameters* p;
//		  Wm = p = m.add_parameters({composedSize, voc.vsize});
		  Wf = p = (feature_mask&USE_WORDWISE) ? m.add_parameters({composedSize, nMiscFeatures}):0;
		  Wd = p = (distPoints&0x1 && feature_mask&USE_DIST) ? m.add_parameters({composedSize, nDistFeatures}):0;
		  Wp = p = m.add_parameters({composedSize, posC});
		  b1 = p = m.add_parameters({composedSize});
		  W2 = p = m.add_parameters({HIDDEN2_SIZE,composedSize });
		  W2d = p = (distPoints&0x2) ? m.add_parameters({HIDDEN2_SIZE, nDistFeatures }):0;
		  b2 = p = m.add_parameters({HIDDEN2_SIZE});
		  W3 = p = m.add_parameters({HIDDEN3_SIZE,HIDDEN2_SIZE });
		  b3 = p = m.add_parameters({HIDDEN3_SIZE});;
		  W4 = p = m.add_parameters({1,HIDDEN3_SIZE });
		  b4 = p = m.add_parameters({1});;
		  W2s = p = m.add_parameters({S_HIDDEN2_SIZE,  composedSize});
		  b2s = p = m.add_parameters({S_HIDDEN2_SIZE});
		  W3s = p = m.add_parameters({senseC,S_HIDDEN2_SIZE });
		  b3s = p = m.add_parameters({senseC});
		  v0 = p = m.add_parameters({composedSize});

		compose_W0 = (feature_mask & RECURSE_COMPOSE) ? m.add_lookup_parameters(posC,{composedSize, composedSize}):0,
		compose_W1 = (feature_mask & USE_WORDVEC) ? m.add_lookup_parameters(posC,{composedSize, voc.vsize}):0;
		compose_Wh = (feature_mask & USE_HASH) ? m.add_lookup_parameters(posC,{composedSize,    HASH_DIM}):0;
		
		if (!initialized) return initialized = true;
		return false;
  }

  Expression classify(Expression l1, Expression d) {
	  ComputationGraph& cg = *l1.pg ;
	  Expression  layer2 = parameter(cg,W2)* l1 + parameter(cg,b2);
	  if (W2d) layer2 = layer2  + parameter(cg,W2d) * d;
	  if (hasNan(& layer2)) cerr << "Achtung ! NAN !" << endl, abort();
	  Expression layer3 =  tanh(parameter(cg,W3) * layer2 + parameter(cg,b3));
	  return (tanh(parameter(cg,W4) * layer3 + parameter(cg,b4)));
  }
  Expression sense(Expression l1) {
	  ComputationGraph&  cg = *l1.pg ;
	  Expression	layer2 = parameter(cg,W2s)* l1 + parameter(cg,b2s);
	  return tanh(parameter(cg,W3s) * layer2 + parameter(cg,b3s));
  }

  float learnOne(Expression layer1, Expression d, bool ground, bool validation) {
	ComputationGraph& cg =  *layer1.pg ;
	Expression g = input(cg,ground? kSCALAR_ONE: kSCALAR_MINUSONE);
	Expression cl = classify(layer1, d);
	Expression loss = squared_distance( cl, g );
	cnn::real incloss = as_scalar(cg.forward());
	if (!validation) cg.backward();
	return incloss;
}

  Expression compose_v(Expression vc1, Expression v2, Expression h2, POS p2, Expression m, Expression f, Expression d) {
		ComputationGraph&  cg = *f.pg ;
		Expression comp_v = parameter(cg,b1);
	  	if (compose_W0) comp_v = comp_v + lookup(cg, compose_W0, p2) * vc1 ;
	  	if (compose_Wh) comp_v = comp_v + lookup(cg, compose_Wh, p2) * h2 ;
	  	if (compose_W1) comp_v = comp_v + lookup(cg, compose_W1, p2) * v2 ;
	  	if (Wd) comp_v = comp_v + parameter(cg,Wd) * d ;
	  	if (Wf) comp_v = comp_v + parameter(cg,Wf) * f ;
	  	
	  return tanh(/*parameter(cg,Wm)* m +*/ comp_v) ;
  }
  
  Expression compose_v1(Expression v2, Expression h2, POS p2, Expression m, Expression f) {
		ComputationGraph&  cg = *f.pg ;
		Expression comp_v = parameter(cg,b1);
	  	if (compose_W0) comp_v = comp_v + lookup(cg, compose_W0, p2) * parameter(cg,v0) ;
	  	if (compose_Wh) comp_v = comp_v + lookup(cg, compose_Wh, p2) * h2 ;
	  	if (compose_W1) comp_v = comp_v + lookup(cg, compose_W1, p2) * v2 ;
	  	if (Wf) comp_v = comp_v + parameter(cg,Wf) * f ;
	  	
	  return tanh(/*parameter(cg,Wm)* m +*/ comp_v) ;
  }

  float learnOneSense(Expression v, Sense s, bool validation) {
	ComputationGraph&  cg = *v.pg ;
	Expression g = oneHot(cg,s,tables.senses.size());
	Expression sense_pred = sense(v);
	Expression loss = squared_distance( sense_pred, g );
	cnn::real incloss = as_scalar(cg.forward());
	if (!validation) cg.backward();
	return incloss/tables.senses.size();
}
	MWCompModel() :initialized(false), tables({initialized, initialized}),
		distPoints(1U), feature_mask(USE_WORDVEC|USE_HASH|USE_DIST|USE_WORDWISE|RECURSE_COMPOSE),
		composedSize(DEFAULT_COMPOSED_SIZE) {}
};

//A structure for each word position in a sequence
struct SeqTok {
	string word;
	string lemma;
	const WordInfo  *lex;
	vector<float> h;
	POS pos;
	char mw;
	int MWparent;
	Sense sense;
	vector<float> score; //for debugging
	vector<cnn::real> misc;
	int pHead;
	bool prequoted, postquoted;
	int nquotes;
	SeqTok(const char* w, const char* l, POS p,
	 char m, int h, Sense s, const MWCompModel& mw_model):
	    word(w),lemma(l),
		lex(&lookup(voc,w,l)), h(bitvec((lex==&unknownWord)?::hash(w):0,HASH_DIM)), 
		pos(p),mw(m),MWparent(h),sense(s),pHead(-1),
		prequoted(false), postquoted(false), nquotes(0)  {}
};

//Reads sentences from a file in DiMSUM CoNLL-like format
void readConll(vector<vector<SeqTok> > & text, MWCompModel& mw_model,   const string& fname) {
	MWCompModel::Tables& tables = mw_model.tables;
	ifstream is(fname);
	string st;
	while (!is.eof()) {
		text.push_back(vector<SeqTok>());
	while (!is.eof()) {
		getline(is,st);
		vector<string> t;
		boost::split(t, st, [](char c) { return c=='\t'; });
		if (t.size() < 8) break;
		text.back().push_back(SeqTok(
			t[1].c_str(),t[2].c_str(),tables.pos[t[3]],
			t[4][0],strtol(t[5].c_str(),0,0) - 1,
			tables.senses[t[7]],
			mw_model) );
		}
	for (size_t i=0; i<text.back().size(); ++i) {
			SeqTok& t= text.back()[i];
			int c;
			//cumulative number of double quotes
			t.nquotes = (c=count(t.word.begin(), t.word.end(), '"')) + (i? text.back()[i-1].nquotes: 0);
			if (c) {
				if (i) text.back()[i-1].postquoted = true;
				if (i<text.back().size()-1) text.back()[i+1].prequoted = true;
			}
		}
	}
	if (wlookuplog.is_open()) wlookuplog<<flush;
}

//Returns a (longest) parse tree distance from two worrds in a sequence
//to their nearest common phrase
int hierDist(const vector <SeqTok> & sent, int j, int k   ) {
	set<int> lots;
#ifdef HIER_DIST_DEBUG
	cerr << "hierDist " << sent[j].word << " [" << sent[j].pHead << "] -- " << sent[k].word << "[" << sent[k].pHead << "] :";
#endif
	int d = 0;
	bool cj;
	while ( cj = j >=0 && lots.insert(j).second && (j = sent[j].pHead, true),
		    k >=0 && lots.insert(k).second && (k = sent[k].pHead, true) && cj )
		++d;
#ifdef HIER_DIST_DEBUG
	cerr << d << endl;
#endif
	return d;
}

//If two words share the same phrase and one of them is the main word of the phrase
bool areParentAndChild(const vector <SeqTok> & sent, int j, int k   ) {
	return (sent[j].pHead==k || sent[k].pHead==j);
}

//Reads a parser output CoNLL format file and adds parser data to an existing text structure
//Parser file should be exactly alighned string by string to the DiMSUM input file
void addParserInfo(vector<vector<SeqTok> > & text,  const string& fname) {
	size_t i = 0;
	ifstream is(fname);
	string st;
	while (!is.eof()) {
		size_t  j=0;
	while (!is.eof()) {
		getline(is,st);
		vector<string> t;
		boost::split(t, st, [](char c) { return c=='\t'; });
		if (t.size() < 8) break;
		if (j>=text[i].size()) {
			if (j==text[i].size())
				cerr << "Parser file " << fname << " mismatch to Multiword Expression file, sentence # " << i << ", token # "<<j<< endl;
		} else
			text[i][j].pHead = strtol( t[6]. c_str(),0,0 ) - 1;
		++j;
	}
		if (++i >= text.  size())  return;
}}

//Read a binary Google Word2Vec format file
//(uses a fragment from distance.c)
bool readVectors(Vocab& vocab,const char *file_name)
{
  FILE *f;
const long long max_size = 2000;         // max length of strings
//  char st1[max_size];
//  char file_name[max_size], st[100][max_size];
  float dist, len;
  long long words, /*size, */ a, b, c, d, cn, bi[100];
  char ch;
  float *M;
//  char *vocab;

  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return false;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%ld", &vocab.vsize);
//  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (b = 0; b < words; b++) {
    a = 0;
    char word[max_size];
    while (1) {
      word[a] = fgetc(f);
      if (feof(f) || (word[a] == ' ')) break;
      if ((a < vocab.vsize) && (word[a] != '\n')) a++;
    }
    word[a] = 0;
    WordInfo& info = vocab.words.emplace(std::piecewise_construct,
              std::forward_as_tuple(word),
              std::forward_as_tuple(b, vocab.vsize)).first->second;

    for (a = 0; a < vocab.vsize; a++) fread(&info.v[a], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < vocab.vsize; a++) len += info.v[a] * info.v[a];
    len = sqrt(len);
    if (len > 0.) for (a = 0; a < vocab.vsize; a++) info.v[a] /= len;
  }
  fclose(f);
  return true;
}

//Choosed a random word position to create a negative sample of an MWE
int chooseRandomOther(const vector<int>& stops, int mwi) {
	int skip_length = stops[mwi] - mwi;
    int fullrange = stops.size() - skip_length;
    int range = min(10,min(fullrange, max((int)stops.size() - stops[mwi], 4)));
	if (range <= 0) return -1;
	int rnd = random()%range;
	int choice = (stops[mwi]+rnd)%stops.size();
	assert(choice >= stops[mwi] || choice < mwi);
	return choice;
}


//A structure for denoting stanalone words and MWE structure in a sequence
//Let i = 0..Z-1, where Z is the number of words in a sentence
//  [ j..stops[j] ) range will correspond to one minimum semantic unit,
//  if initially j=0 and each next iteration j=stops[j]
//  and a 'real' word position for i in the sentence will be idx[i] 
struct SentenceMWs  {
	vector <int> stops;
	vector <int> idx;
};

//Returns the semantic unit head word position for the i_th position in a sentence
inline int MWstart(const vector<SeqTok>& s, int i) {
	while (s[i].MWparent >= 0) i = s[i].MWparent;
	return i;
}

//Fills a SentenceMWs structure for a sequence with minimum semantic units bounary denotions
void groupMWs(vector<int>& outIdx, vector<int>& stops, const vector<SeqTok>& s ) {
	vector<pair<int,int> > sorter(s.size());
	outIdx.resize(s.size()); stops.resize(s.size());
	for (size_t i=0; i < s.size(); ++i) sorter[i]=pair<int,int>(MWstart(s, i), i);
	sort(sorter.begin(),sorter.end());
	for (size_t i=0; i < s.size(); ++i) outIdx[i] = sorter[i].second;
	for (size_t i=s.size(); i > 0; --i) stops[i-1] = (i < s.size() && sorter[i-1].first == sorter[i].first) ? stops [i] : i;
}

void groupMWs(vector<SentenceMWs> &mws, const vector<vector<SeqTok> > &s ) {
	mws.resize(s.size());
	for (size_t i=0; i<s.size(); ++i)
		groupMWs(mws[i].idx, mws[i].stops, s[i]);
}

//Calculate mean word vectors for sentences
void calcMean(vector<vector<float> >& mean, const vector<vector<SeqTok> >& s ) {
	mean.resize(s.size());
	for (size_t i=0; i<s.size(); ++i ) if (s[i].size() > 0) {
		mean[i]=s[i][0].lex->v;
		for (size_t j=1; j<s[i].size(); ++j)
		for (size_t k=0; k<mean[i].size(); ++k) {
			mean[i][k]+=s[i][j].lex->v[k];
		}
		for (size_t k=0; k<mean[i].size(); ++k) {
			mean[i][k]/=s[i].size();
		}
	}
}

//Calculate a word-wise vector of heuristic features
void miscFeatures(vector<float>& fi, const SeqTok& word, int ncapital) {
	fi.clear();
	bool capital = isCapital(word.word.c_str()),
		isNum = isNumber(word.word.c_str()),
		hasNum = hasNumber(word.word.c_str());
	float capital_norm = capital? 1. / ncapital : 0.;
	float uprRatio = upperRatio(word.word.c_str() );
	bool isUrl = !strcasecmp(word.word.c_str(),"URL") || strstr(word.word.c_str(),"://");
	bool isHash = word.word[0]=='#';
	bool isAt = word.word[0]=='@';
//
	bool hasPrime = strchr(word.word.c_str(),'\'');
	bool isPunct = false;
	static const char* punct= "!?.,;:{}[]()/";
	for (const char* p = punct; *p; ++p) if (strchr(word.word.c_str(), *p)) isPunct = true;
	bool hasNonAlpha = hasNonAlphaChar(word.word.c_str());

	fi.push_back(plusminusbit(isNum));
	fi.push_back(plusminusbit(hasNum));
	fi.push_back(plusminusbit(capital));
	fi.push_back(plusminusbit(hasPrime));
	fi.push_back(plusminusbit(isPunct));
	fi.push_back(plusminusbit(hasNonAlpha));
	fi.push_back(plusminusbit(isUrl));
	fi.push_back(plusminusbit(isHash));
	fi.push_back(plusminusbit(isAt));
	fi.push_back(plusminusbit(word.prequoted));
	fi.push_back(plusminusbit(word.postquoted));
	fi.push_back(uprRatio);
	fi.push_back(capital_norm);
	fi.push_back(log(word.lex->range+10)-log(10));
	fi.push_back(plusminusbit(word.lex ==  &unknownWord));
	assert(fi.size()==nMiscFeatures);
}

const char* miscFeatureChart[nMiscFeatures] = {
	"is_num",
	"has_num",
	"1st_cap",
	"has_prime",
	"is_punct",
	"has_non_alpha",
	"is_url",
	"is_hash",
	"is_at",
	"prequoted",
	"postquoted",
	"cap_ratio",
	"cap_norm",
	"log_range",
	"is_unknown"
};

//Calculates an inter-word distance feature vector
//(that includes parser-based phrase distances)
void distFeatures(vector<float>& fi, const vector<SeqTok>& sent, int jp, int j) {
	if (jp > j) {int tmp=j; j=jp; jp=tmp;} //now j >= jp
	fi.clear();
	bool interquote = sent[j-1].nquotes!=sent[jp].nquotes;
	fi.push_back((j-jp-1)/8.);
	switch (hierDist(sent,jp,j)) {
		case 1: case 0 /*never happens*/: fi.push_back(2.); break;
		case 2: fi.push_back(0.); break;
		default: fi.push_back(-1.5); break;
	}
	fi.push_back(plusminusbit(areParentAndChild(sent,jp,j)));
	fi.push_back(plusminusbit(interquote));
	
	assert(fi.size()==nDistFeatures);
}

const char* distFeatureChart[nDistFeatures] = {
	"gap",
	"par_dist",
	"par_parent",
	"inter_quote"
};

//Learns an MWE+Supersense model from an nnotated text structure
pair<float,float>   learnAll(const vector<vector<SeqTok> >& s, MWCompModel& mw_model,
 	Trainer& sgd,  bool validation) {
	vector<vector<float> > mean;
	vector<SentenceMWs>  aux;
	int MW = 0,   longMW = 0;
	calcMean(mean, s);
	groupMWs(aux, s);
	assert (s.size()==mean.size());
	assert (s.size()==aux.size());

	pair<float, float>loss (0.,0.);
	float oneLoss;
 	vector <int> pick(s.size());
 	for (int ii=0; ii<s.size(); ++ii) pick[ii]=ii;
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	time_t ti;
	shuffle (pick.begin(), pick.end(), std::default_random_engine(time(&ti)));

 	//Randomly iterate sequences
 	for (int ii=0; ii<s.size(); ++ii) {
		int i = pick[ii] ;
		ComputationGraph cg;
		list<vector<float> > dvs;
		Expression m = input(cg, {voc.vsize},&mean[i]);
		//Iterate through minimum semantic unit head words
		for (size_t j=0; j<s[i].size(); j=aux[i].stops[j]) {
		  int true_j = aux[i].idx[j];
		  POS  p =  s[i][true_j].pos;
		  Expression f = input(cg,{nMiscFeatures},&s[i][true_j].misc),
				v = mw_model.compose_v1 (
					input(cg,{voc.vsize},&s[i][true_j].lex->v),input(cg,{HASH_DIM},&s[i][true_j].h),p,m,
					f);
		  int k = j, true_k = true_j;
		  //An inner loop: iterate through true and false MWE extension samples
		  while (true) {
			//This loop is reserved for multiple negative samples option
			for (int n=0; n<1; ++n) {
			  //A false sample - Non Multi word
			  int z = chooseRandomOther(aux[i].stops, j) ;
			  if (z >= 0) {
				int true_z = aux[i].idx[z];
				if(!( true_z > true_k && true_z < true_k + 10)) continue;
				dvs.emplace_back();
				vector<float>& dv = dvs.back();
				distFeatures(dv,s[i],true_z,true_k);
				POS pz = s[i][true_z].pos;
				Expression vz = input(cg, {voc.vsize},&s[i][true_z].lex->v), hz = input(cg,{HASH_DIM},&s[i][true_z].h),
					fz = input(cg,{nMiscFeatures},&s[i][true_z].misc),
					d = input(cg,{nDistFeatures},&dv),
					comp_v = mw_model.compose_v (v,vz,hz,pz,m,fz,d) ;
				loss.first+=oneLoss = mw_model. learnOne(comp_v, d, false, validation);
				corr.add(MW_LOG_ID, false, f,miscFeatureChart,fz,miscFeatureChart,d,distFeatureChart);
#ifdef DEBUG_MW_SEL
				cout << "Non-MW##"<<true_k<<", "<<true_z<<" = "<<oneLoss<<"\t";
#endif
#ifdef LOG_FEATURE_VALUES
				logVectors(vf, 'F', 0, &v,&vz,&hz,&comp_v,&m,&fz,&d);
#endif
				if (!validation) sgd.update(rate);
			  }
			}			
			++k;
			if (k>=aux[i].stops[j]) break; //inner loop exit here - end of minimum semantic unit
			++MW; if (k>=j+2) ++longMW;
			true_k = aux[i].idx[k];
			//Mutiword extension! (note: modifies h and v so keep it after 
			//a negative sample processing
			dvs.emplace_back();
			vector<float>& dv = dvs.back();
			distFeatures(dv,s[i],aux[i].idx[k-1]/*prev true_k*/,true_k);
			POS pk = s[i][true_k].pos;
			Expression vk = input(cg,{voc.vsize}, &s [i][true_k].lex->v), hk = input(cg,{HASH_DIM},&s[i][true_k].h),
				fk = input(cg,{nMiscFeatures},&s[i][true_k].misc),
				d = input(cg,{nDistFeatures},&dv) ,
				comp_v = mw_model.compose_v  (v,vk,hk,pk,m,fk,d);
			loss.first += oneLoss = mw_model.learnOne(comp_v, d, true, validation); 
			corr.add(MW_LOG_ID,true,  f,miscFeatureChart,fk,miscFeatureChart,d,distFeatureChart );
#ifdef DEBUG_MW_SEL
			cout << "MW##"<<true_j<<", "<<true_k<<" = "<<oneLoss<<"\t";
#endif
#ifdef LOG_FEATURE_VALUES
			logVectors(vf, 'T', 1, &v,&vk,&hk,&comp_v,&m,&d);
#endif
			if (!validation) sgd.update(rate);
			v=comp_v;
		  }

			
		  //Now comp_v and comp_h contain combined senses for Multi word
		  loss.second += oneLoss = mw_model.learnOneSense(v,s[i][true_j].sense, validation);
		  for (size_t sen=0; sen < mw_model.tables.senses.size(); ++sen)
			 corr.add( string("sense:")+mw_model.tables.senses.at(sen),s[i][true_j].sense==sen, f,miscFeatureChart);
#ifdef DEBUG_MW_SENSE
		  cout << "Learning sense of word #"<<true_j<<", loss = "<<oneLoss<<endl;
#endif
#ifdef LOG_FEATURE_VALUES
		  logVectors(vsf, 'S',  s[i][true_j].sense, &v);
#endif
		  sgd.update(rate);

		}
		cout << "\n" << (validation? "[V] ":"") <<  ii <<". Sentence " << i<<" , loss = " << loss.first << " , senseLoss = " << loss.second << endl;
		cerr << "MWs processed: " << MW << ", incl. Long MWs: " << longMW << endl;
	}
	if (corr.accepting) {corr>>correlation_tab; correlation_tab.close(); corr.accepting=false;}
	return loss;
}

//Calculate a word-wise features throughout the text
void calcMisc(vector<vector<SeqTok> > &s ) {
	for (size_t i=0; i<s.size();  i++) {
				  int ncap = std::accumulate(s [i].begin(), s[i].end(), 0, [](int n, const SeqTok& tok) {return isCapital(tok.word.c_str())?n+1:n;});
			for (size_t j=0; j<s[i].size(); ++j) miscFeatures(s[i][j].misc, s[i][j], ncap   );
	}
}

//Predicts MWE boundries and Supersenses for a text structure
void predict (vector<vector<SeqTok> >& s, MWCompModel& mw_model) {
	vector<vector<float> > mean;
	calcMean(mean,s);
	//An outer loop: a minimum semantic unit head word
	for (size_t i=0; i<s.size(); ++i) {
		cerr << i << ".\r"  ;
		//clear MWE info
		size_t vol = s[i].size();
		bool consumed[vol], inside[vol];
		fill(consumed,	consumed+vol,false);
		fill(inside,	inside+vol,false);
		for (size_t j =0; j<vol; ++j) {
				s[i][j].MWparent  = 0;
				s[i][j].mw =  'O';
		}
		for (size_t m=0; m<vol; ++m) if (!consumed[m]) {
				ComputationGraph cg;
				list<vector<float> > dvs;
				POS p = s[i][m].pos;
				Expression mn =  input(cg, {voc.vsize},&mean[i]), 
				v = mw_model. compose_v1(
					input(cg,{voc.vsize},&s[i][m].lex->v),
					input(cg,{HASH_DIM},&s[i][m].h),p,mn,
					input(cg,{nMiscFeatures},&s[i][m].misc));
			size_t inside_start = m+1;

								
			size_t last_mw_j = m;
			//an inner loop: possible word to continue an MWE
			for (int j=m+1; j<min(vol,last_mw_j+10)&&!consumed[j]; ++j) {
				dvs.emplace_back();
				vector<float>& dv = dvs.back();
				distFeatures(dv, s[i], last_mw_j, j);
				POS pj = s[i][j].pos;
				Expression vj = input(cg,{voc.vsize},&s[i][j].lex->v), hj = input(cg,{HASH_DIM},&s[i][j].h),
					fj = input(cg,{nMiscFeatures},&s[i][j].misc),
					d = input(cg,{nDistFeatures},&dv),
					comp_v = mw_model.compose_v (v,vj,hj,pj,mn,fj,d);
				mw_model.classify(comp_v, d);
				float as1 = as_scalar(cg.forward()), conf = 0;
				s[i][j].score.push_back(as1);
				//s[i][j].score.push_back(as2);
				if (hierDist(s[i], last_mw_j, j) <= 1 && as1 > conf   ) {
					//j_th word is predicted to be a Multiword Expression extension
					consumed [j] = true;
					v = comp_v;
					s[i][m].mw = inside[m]?  'b': 'B';
					s[i][j].MWparent = m+1 ;
					s[i][j].mw = inside[m]?  'i': 'I';
					while(inside_start < j)  {
						if (s[i][inside_start].mw=='o') {
							cerr << "Warning: gap inside a predicted MWE filled due to format restrictions for word "
								 << inside_start << ": " << s[i][inside_start].word<<endl;
							s[i][inside_start].mw= 'i';
							s[i][inside_start].MWparent= m+1 ;
							}
						else if (s[i][inside_start].mw=='O') { 
						  	s[i][inside_start].mw= 'o';}
						inside[inside_start++]=true;}
					last_mw_j = j;
				}
			}
			mw_model.sense(v);
			vector<float> vs(as_vector(cg.forward()));
			s[i][m].sense = max_element(vs.begin(),vs.end())-vs.begin() ;
		}
	}
	cerr <<"\n";
}

//Download sentences with predicted data in DiMSUM Conll format
void writeConll(const vector<vector<SeqTok> >& s, MWCompModel::Tables& tables, const string& f, bool debugScore) {
	ofstream of(f);
	for (size_t i=0; i<s.size(); ++i) {
		for(size_t j=0; j<s[i].size(); ++j) {
			const SeqTok &tok = s[i][j];
			of << j+1 << "\t" << tok.word << "\t" <<  tok.lemma << "\t" << tables.pos.at(tok.pos) << "\t" <<
				tok.mw << "\t" << tok.MWparent << "\t";
			if (debugScore) {for (size_t k=0; k<tok.score.size(); ++k) of << tok.score[k] <<"/";}
			of << "\t"  << tables.senses.at(tok.sense) << endl;
				
		}
		of << endl;
	}
  }

//Produces a wide sheet of word features for each word as well as a list of inter-word
//feature values for -spread..+spread inter-word offsets. May be used to export feature
//values into an external classifier.  Invoked in -y mode
void writePairFeats(vector<vector<SeqTok> >& s, MWCompModel& mw_model, size_t spread, const string& f  ) {
	//Format note
	int N = 2* nMiscFeatures + nDistFeatures + voc.vsize + 1 + mw_model.tables.senses.size();
	cerr << "File row format: " << spread <<
		"x Pairwise_Feats block (for left_word: this position, right word: (this + (1.."<<spread<<")) position" << endl <<
		"	(filled with zeros if right word position is outside the sentence)" << endl <<
		"Pairwise_Feats := " << 
		"\t" <<nMiscFeatures << " misc_feats for the left word + " << endl <<
		"\t" << nMiscFeatures << " misc_feats for the right word + " << endl <<
		"\t" << nDistFeatures << " distance_feats + " << endl <<
		"\t" << voc.vsize << " composite vector coorinates + " << endl <<
		"\t" << 1 << " MW classifier output + " << endl <<
		"\t" << mw_model.tables.senses.size() << " sense classifier output" << endl <<
		"(" << N << " numbers in total per Pairwise_Feats block, "<<
		N*spread << " numbers per line)" << endl;
;
	ofstream os(f);
	vector<vector<float> > mean;
	calcMean(mean,s);
	for (size_t i=0; i<s.size(); ++i) {
			cerr << i << ".\r";

		for (size_t m=0; m<s[i].size(); ++m) {
				ComputationGraph cg;
				list<vector<float> > dvs;
				POS p = s[i][m].pos;
				Expression mn =  input(cg, {voc.vsize},&mean[i]),
					v = mw_model.compose_v1(input(cg,{voc.vsize},&s[i][m].lex->v), input(cg,{HASH_DIM},&s[i][m].h),p,mn,
							input(cg,{nMiscFeatures},&s[i][m].misc));
					
			for (int j=m+1; j<=m+spread; ++j) {
				if (j >= s[i].size()) {
					for (size_t y=0; y<N; ++y) os << "0\t";
					continue;
				}
				dvs.emplace_back();
				vector<float>& dv = dvs.back();
				distFeatures(dv, s[i], m, j);
				POS pj = s[i][j].pos;
				Expression vj = input(cg,{voc.vsize},&s[i][j].lex->v), hj = input(cg,{HASH_DIM},&s[i][j].h),
					fj = input(cg,{nMiscFeatures},&s[i][j].misc),
					d = input(cg,{nDistFeatures},&dv),
					comp_v = mw_model.compose_v (v, vj,hj,pj,mn,fj,d),
				cm = mw_model.classify(comp_v, d),
				cs = mw_model.sense(v);
				pExp(os, &fj);
				pExp(os, &d);
				pExp(os, &comp_v);
				pExp(os, &cm);
				pExp(os, &cs);
			
			}
			os << "\n";
	}
	os << endl;
	}
	cerr <<"\n";
}

void initialize(MWCompModel &model, const string &filename)
{
 cerr << "Initialising model parameters from file: " << filename << endl;
 ifstream in(filename);
 boost::archive::text_iarchive ia(in);
 ia >> model.tables.pos;
 ia >> model.tables.senses;
 ia >> model.distPoints >> model.composedSize;

 model.initialize();
 ia >> model.m;
}


void store(const MWCompModel &model, const string &fname) {
cout << "# saving the model ... "  << flush;
      ofstream out(fname);
      boost::archive::text_oarchive oa(out);
      oa << model.tables.pos;
      oa << model.tables.senses;
      oa << model.distPoints << model.composedSize;
      oa << model.m;
      cout << " now  done!" << endl;
}

//Splits a training set of sentences into in order to get a validation set
template <typename T> size_t devSplit(vector <T>& orig, vector <T>& dev, size_t devSz,
	const string & mapFile) {
	dev.resize(devSz);
	if (!devSz) return 0U;
	time_t ti;
	vector<bool> pick(orig .size(), false);
	fill (pick.end() - devSz, pick.end(), true);
	shuffle (pick.begin(), pick.end(), std::default_random_engine(time(&ti)));
	ofstream os(mapFile);
	for (size_t i=0; i< orig.size(); ++i) os << pick[i] << "\n";
	for (size_t i=0, j=0, dj=0; i< orig.size(); ++i) 
		if (pick[i]) dev[dj++] = move(orig[i]);
		else {if (i!=j) orig[j] = move(orig[i]); j++;}
	orig.resize(orig.size()-devSz);
	return devSz;
}

int main(int argc, char** argv) {
  int opt;
  bool continueLearn(false), debugScore(false);
  //temporary for old model loading
  size_t spread = 9;
  string tFile, lFile, fFile, pFile, ptFile, plFile, vFile, mFile;
  MWCompModel mw_model;
  while ((opt = getopt (argc, argv, "dD:b:V:%:y:s:v:L:l:T:t:cm:p:")) != -1)
  {
    switch (opt)
    {
      case 'd':
                printf ("Debug mode (show prediction score at prediction file)\n"       );
                debugScore = true;
                wlookuplog.open("classify.lookup.log");
                printf ("Lookup debug file: classify.lookup.log\n");
                corr.accepting=true;
                correlation_tab.open("classify.correlation.tab");
                printf ("Feature correlation file: classify.correlation.tab\n");
                break;
      case 'V':
                mw_model.composedSize = strtol(optarg,0,0);
                printf ("Composed vector size: \"%d\"\n", mw_model.composedSize);
                break;
      case 's':
                printf ("Word position spread: \"%s\"\n", optarg);
                spread = strtol(optarg,0,0);
                break;
      case 't':
                printf ("Test file: \"%s\"\n", optarg);
                tFile = optarg;
                break;
      case 'T':
                printf ("Test parse file: \"%s\"\n", optarg);
                ptFile = optarg;
                break;
      case 'l':
                printf ("Training file: \"%s\"\n", optarg);
                lFile = optarg;
                break;
       case 'L':
                printf ("Training parse file: \"%s\"\n", optarg);
                plFile = optarg;
                break;
      case 'm':
                printf ("Model file: \"%s\"\n", optarg);
                mFile = optarg;
                break;
      case 'v':
                printf ("Embedding Vector file: \"%s\"\n", optarg);
                vFile = optarg;
                break;
      case 'p':
                printf ("Prediction file: \"%s\"\n", optarg);
                pFile = optarg;
                break;
      case 'y':
                printf ("Feature output file: \"%s\"\n", optarg);
                fFile = optarg;
                break;
      case 'c':
                printf ("Continuing learn\n");
                continueLearn = true;
                break;
      case '%':
                devRatio = strtod(optarg,0);
                mw_model.distPoints = printf ("Dev set size ratio : %f\n", devRatio);
                break;
      case 'D':
                mw_model.distPoints = strtol(optarg,0,0);
                printf ("Distance Features insertion point(s): %x (bit mask, 1- composite vector, 2- MWE predictor (layer 1)\n",mw_model. distPoints);
                break;
      case 'b':
                mw_model.feature_mask = strtol(optarg,0,0);
                printf ("Feature Mask: %x (bit mask, 1- word vector, 2- hash, 4- distance, 8-wordwise feats, 0x10- composite recursal\n",mw_model. feature_mask);
                break;
    }
  }

  	cnn::Initialize(argc, argv);

	if (vFile.empty()) 
		cerr << "****** Warning! *******\n"
		" You've started classify without Word Vectors (-v file)\n"
		" If you've done it intentionally pls don't forget to use the same mode at prediction time" << endl;
	else
		readVectors(voc, vFile.c_str());
#ifdef LOOKUP_CORRECTION
	voc.initDerived();
#endif
	unknownWord.v.resize( voc.vsize, 0. );
	if (lFile.empty() || continueLearn) {
		if (mFile.empty()) 
		{
			cerr << "No model specified. Please set either a CoNLL file to learn from (-l filename) or a previously stored model (m filename)" << endl;
			return 1;
		}
			else initialize(mw_model, mFile);
			mw_model.initialized = true;
	}
	if (!lFile.empty()) try {
		vector<vector<SeqTok> > train_s, dev_s;
		readConll(train_s, mw_model, lFile.c_str());
		if (!plFile.empty()) {
			addParserInfo(train_s,plFile);
		}
		if (!mw_model.initialized) mw_model.initialize();
		calcMisc( train_s )   ;
#if  HAVE_CUDA
		SimpleSGDTrainer sgd(&mw_model.m);
#else
		AdadeltaTrainer sgd(&mw_model.m);
#endif
		//SimpleSGDTrainer sgd(&mw_model.m);
		float minLoss = 1e50;
		devSplit(train_s, dev_s, round(devRatio* train_s.size()), mFile+"_devmap");
		for (size_t epoch = 0; epoch < NEPOCH; ++epoch) {
			pair<float,float > loss = learnAll(train_s, mw_model, sgd, false);
			if (!epoch) {ofstream o ("classify.corr"); corr >> o;}
			cout << "Epoch " << epoch<<" , loss = " << loss.first << " , senseLoss = " << loss.second << endl;
			sgd.update_epoch();
			if (dev_s.size()) {
				loss = learnAll(dev_s, mw_model, sgd,true);
				cout << "Epoch " << epoch<< " validation , loss = " << loss.first << " , senseLoss = " << loss.second << endl;
			}
			if (!mFile.empty() && loss.  first + loss.second <minLoss) { 
				store(mw_model, mFile);
				minLoss = loss.first + loss.second ;
			}
		}
		} catch (exception e) {cerr << e.what() << endl;}

	if (!tFile.empty() || !fFile.empty()) try {
		vector<vector<SeqTok> > test_s;
		readConll(test_s, mw_model, tFile);
		if (!ptFile.empty()) {
			addParserInfo(test_s, ptFile);
		}
		calcMisc(test_s);
		if (!pFile.empty()) {
			predict(test_s, mw_model);
			writeConll(test_s, mw_model.tables, pFile, debugScore );
		}
		if (!fFile.empty()) writePairFeats(test_s, mw_model, spread, fFile);
	} catch (exception e) {{cerr << e.what() << endl;}}
	
	return 0;
}


