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

//    #define DEBUG_MW_SEL 1

using namespace std;
using namespace cnn;
using namespace cnn::expr;

const int nMiscFeatures = 9;
const int nDistFeatures = 4 ;

const float rate  = 0.15;
const int NEPOCH = 10024;
const int HIDDEN_SIZE = 256;
const int HIDDEN2_SIZE = 128;
const int S_HIDDEN_SIZE = 128;
const int S_HIDDEN2_SIZE = S_HIDDEN_SIZE;

static int MW = 0;
static int longMW = 0;

ostream& pExp(ostream& o, Expression * x, size_t  limit) {
	vector<float>a (as_vector(x->value() ));
	for (size_t i = 0; i < min(a.size(), limit); ++i) o << a[i] << "\t";
	return o;
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


set<long>cons[2];
int checkConsistency(long i, long true_j, long true_k, bool res)
{
	if (cons[!res].find(i+10000*true_j+1000000*true_k)!=cons[!res].end()) {
		cerr << "Inconsistency for " << i << ", " << true_j << ", " << true_k << endl;
		exit (20);
	}
	cons[res].insert(i+10000*true_j+1000000*true_k);
}
cnn::real plusminusbit(bool v) {
	return v? 1.:-1.;
}

ofstream vf("mw.vec");
ofstream vsf("sense.vec");

vector<cnn::real> bitvec(unsigned mi, size_t  sz) {
      vector<cnn::real> r(sz);
      for (unsigned i = 0; i < sz; ++i) {
      		bool x = (((mi >> i) & 1U) != 0);
      		r[i] = x ? 1 : -1;
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
unsigned long
hash(const char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = (unsigned char) *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

Expression oneHot(ComputationGraph& cg, size_t i, size_t sz) {
	vector<Expression> v(sz,input(cg,kSCALAR_MINUSONE));
	v[i] = input(cg,kSCALAR_ONE);
	return concatenate(v);
}

Expression misc_compose(Expression h, Expression h2) {return (2*h + h2)/3;}
Expression hash_compose(Expression h, Expression h2) {return (h + h2)/2;}

long int HASH_DIM = 16;

typedef size_t Sense,POS;
struct WordInfo {
	WordInfo(unsigned long long int r, size_t dim):range(r),v(dim,0.){}
	long long int range;
	vector<float> v;
};

WordInfo unknownWord = WordInfo(ULLONG_MAX,0U);

struct Vocab {
	long vsize;
	unordered_map< string, WordInfo > words;
	//Derived
	 set<char> smallLetters;
	//end derived
	void initDerived() {
		auto it=words.begin();
		size_t i = 0;
		for (; i<1000 && it!=words.end(); ++i,++it) {
			for (size_t j=0; j<it->first. length(); ++j) {
				smallLetters.insert(it->first[j]);
			}
		}
	}
};

Vocab voc;

inline void skipTagChar(const char* &word) {while (word[0] && strchr("#@",word[0]) ) word++;}
inline bool isCapital(const char* word) {
	skipTagChar(word);
	return isupper(*word);
}
inline float upperNextRatio(const char* word) {
	skipTagChar(word);
	size_t u=0, l=0;
	for (;*word;++word) if (isupper(*word)) ++u; else if (islower(*word)) ++l;
	return u? ((float) u)/(u+l): 0.;
}


inline const WordInfo& lookup(const Vocab& voc, const char* word, const char* lemma) {
	skipTagChar(word);
	auto i = voc.words.find(word);
	if (i != voc.words.end())  {if (hasNan(i->second.v)) cerr << "NaN word: " << word << endl; return i->second; }
	string w(word) ;
	std::transform(w.begin(), w.end(), w.begin(), ::tolower);
	i = voc.words.find(word);
	if (i != voc.words.end()) {if (hasNan(i->second.v)) cerr << "NaN word: " << word << endl; return i->second; }
	for (size_t pos=0; pos<w.length();pos++) {
		char old = w[pos];
		for (auto l:voc.smallLetters) if (l!=old) {
			w[pos]=l;
			auto j = voc.words.find(word);
			if (j!=voc.words.end() && (i==voc.words.end() || j->second.range < i->second.range) ) i=j;
			}
		w[pos]=old;
	}
	if (i != voc.words.end()) {if (hasNan(i->second.v)) cerr << "NaN word: " << word << endl; return i->second; }
	cerr << "Unknown word: " << word << endl;
	return unknownWord;
}

struct UnexpectedEntityException: public exception {
	string msg;
	UnexpectedEntityException(const string &name) : msg(string("New entity encountered while model is closed: ")+name) {}
	virtual const char* what() const noexcept {return msg.c_str();  }
};

void serialize(boost::archive::text_oarchive& oa, const string& s, unsigned int ver) {oa & s;}
void serialize(boost::archive::text_iarchive& ia, string& s, unsigned int ver) {ia >> s;}
	
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

struct SeqTok {
	string word;
	string lemma;
	const WordInfo  *lex;
	vector<float> h;
	POS pos;
	char mw;
	int MWhead;
	Sense sense;
	vector<cnn::real> misc;
	int pHead;
	bool prequoted, postquoted;
	int nquotes;
	SeqTok(const char* w, const char* l, POS p, char m, int h, Sense s):word(w),lemma(l),lex(&lookup(voc,w,l)),h(bitvec(::hash(w),HASH_DIM)), 
		pos(p),mw(m),MWhead(h),sense(s),pHead(-1),
		prequoted(false), postquoted(false), nquotes(0)  {}
};

//This procedure is based on a fragment of code
//taken from http://stackoverflow.com/questions/236129/split-a-string-in-c
 std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
//        trim(item);
        if (!item.empty()) elems.push_back(item);
    }
    return elems;
}

/* void collect(map<string,int> p, int i, const string& fname) {
	cerr << "Collecting field "<<i<<" :"<<  endl;
	ifstream is(fname);
	string st;
	while (!is.eof()) {
		text.push_back(vector<SeqTok>());
	while (!is.eof()) {
		getline(is,st);
		vector<string> t;
		boost::split(t, st, [](char c) { return c=='\t'; });
		if (t.size() < 8) break;
		if (p.insert(pair<string,int>(t[i],p.size()).first)
				cerr << " -  "<<t[i]<<endl;;
		}
	}
}
*/
struct MWCompModel {
bool initialized;
struct Tables  { NameHash pos, senses ; } tables;

    LookupParameters *compose_W0, *compose_W1;
	Parameters * Wv1, * Wv2, * Wh1, * Wh2, * Wx, * W2, * W3, * Wm, * Wf, * Wd, * b1, * b2, * b3,
		* Wvs, * Whs, * W2s, * W3s, * b1s, * b2s, * b3s;
	Model m;

	bool initialize() {
	  return initialize(this->tables.pos.size(), this->tables.senses.size());
	}
	bool initialize(int posC, int senseC) {
	  Parameters* p;
	  	  Wh1 = p = m.add_parameters({HIDDEN_SIZE, HASH_DIM});
	  	  Wh2 = p = m.add_parameters({HIDDEN_SIZE, HASH_DIM});
		  Wm = p = m.add_parameters({HIDDEN_SIZE, voc.vsize});
		  Wx = p = m.add_parameters({HIDDEN_SIZE, voc.vsize});
		  Wf = p = m.add_parameters({HIDDEN_SIZE, nMiscFeatures});
		  Wd = p = m.add_parameters({HIDDEN_SIZE, nDistFeatures});
		  Wv1 = p = m.add_parameters({HIDDEN_SIZE, voc.vsize});
		  Wv2 = p = m.add_parameters({HIDDEN_SIZE, voc.vsize});
		  b1 = p = m.add_parameters({HIDDEN_SIZE});
		  W2 = p = m.add_parameters({HIDDEN2_SIZE,HIDDEN_SIZE });
		  b2 = p = m.add_parameters({HIDDEN2_SIZE});
		  W3 = p = m.add_parameters({1,HIDDEN2_SIZE });
		  b3 = p = m.add_parameters({1});;
	  	  Whs = p = m.add_parameters({S_HIDDEN_SIZE, HASH_DIM});
		  Wvs = p = m.add_parameters({S_HIDDEN_SIZE, voc.vsize});
		  b1s = p = m.add_parameters({S_HIDDEN_SIZE});
		  W2s = p = m.add_parameters({S_HIDDEN2_SIZE, S_HIDDEN_SIZE });
		  b2s = p = m.add_parameters({S_HIDDEN2_SIZE});
		  W3s = p = m.add_parameters({senseC,S_HIDDEN2_SIZE });
		  b3s = p = m.add_parameters({senseC});
		  

		compose_W0 = m.add_lookup_parameters(posC*posC,{voc.vsize, voc.vsize}),
		compose_W1 = m.add_lookup_parameters(posC*posC,{voc.vsize, voc.vsize});
		
		if (!initialized) return initialized = true;
		return false;
  }
	
	Expression classify(Expression v1, Expression h1, Expression v2, Expression h2, Expression comp_v, Expression comp_h, Expression m,
		Expression f, Expression d) {
	  ComputationGraph& cg = *v1.pg ;
	  Expression	layer1 = tanh(parameter(cg,Wm)* m  + parameter(cg,Wv1)* v1 + parameter(cg,Wv2)* v2+ /*parameter(cg,Wx)*comp_v +  */
	   /*parameter(cg,Wh1)* h1 +*/ parameter(cg,Wh2)* h2 + parameter(cg,Wf) * f + parameter(cg,Wd) * d
	   + parameter(cg,b1)),
	  layer2 = parameter(cg,W2)* layer1 + parameter(cg,b2);
	  if (hasNan(& layer2)) cerr << "Achtung ! NAN !" << endl, abort();
	  return (tanh(parameter(cg,W3) * layer2 + parameter(cg,b3)));
  }
	Expression sense(Expression v) {
	  ComputationGraph&  cg = *v.pg ;
	  Expression	layer1 = tanh(parameter(cg,Wvs) *v),
	  layer2 = parameter(cg,W2s)* layer1 + parameter(cg,b2s);
	  return tanh(parameter(cg,W3s) * layer2 + parameter(cg,b3s));
  }

  float learnOne(Expression v1, Expression h1, Expression v2, Expression h2, Expression comp_v, Expression comp_h, Expression m, Expression f, Expression d, bool ground) {
	ComputationGraph& cg =  *v1.pg ;
	Expression g = input(cg,ground? kSCALAR_ONE: kSCALAR_MINUSONE);
	Expression cl = classify(v1, h1, v2, h2, comp_v, comp_h,m,f,d);
	Expression loss = squared_distance( cl, g );
	cnn::real incloss = as_scalar(cg.forward());
	cg.backward();
	return incloss;
}

  Expression compose_v(Expression v1, POS p1, Expression v2, POS p2) {
	  ComputationGraph&  cg = *v1.pg ;
	  Expression l1 = lookup(cg, compose_W0, p1+p2*tables.pos.size()),
		r1 = l1*v1,
	  	l2 = lookup(cg, compose_W1, p1+p2*tables.pos.size()),
	  	r2 = l2*v2,
	  	r = r1 + r2;
	  return r;
  }
  
  float learnOneSense(Expression v, Sense s) {
	ComputationGraph&  cg = *v.pg ;
	Expression g = oneHot(cg,s,tables.senses.size());
	Expression sense_pred = sense(v);
	Expression loss = squared_distance( sense_pred, g );
	cnn::real incloss = as_scalar(cg.forward());
	cg.backward();
	return incloss/tables.senses.size();
}
	MWCompModel() :initialized(false), tables({initialized, initialized}) {}
};
void readConll(vector<vector<SeqTok> > & text, MWCompModel::Tables& tables,   const string& fname) {
	ifstream is(fname);
	string st;
	while (!is.eof()) {
		text.push_back(vector<SeqTok>());
	while (!is.eof()) {
		getline(is,st);
		vector<string> t;
		boost::split(t, st, [](char c) { return c=='\t'; });
		if (t.size() < 8) break;
		text.back().push_back(SeqTok(t[1].c_str(),t[2].c_str(),tables.pos[t[3]],t[4][0],strtol(t[5].c_str(),0,0) - 1, tables.senses[t[7]]));
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
}
int hierDist(const vector <SeqTok> & sent, int j, int k   ) {
	set<int> lots;
	int d = 0;
	bool cj;
	while ( cj = j >=0 && lots.insert(j).second && (j = sent[j].pHead, true),
		    k >=0 && lots.insert(k).second && (k = sent[k].pHead, true) && cj )
		++d;
	return d;
}

bool areParentAndChild(const vector <SeqTok> & sent, int j, int k   ) {
	return (sent[j].pHead==k || sent[k].pHead==j);
}

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

/* uses a fragment from distance.c*/
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

int chooseRandomOther(const vector<int>& stops, int mwi) {
	int skip_length = stops[mwi] - mwi;
    int range = stops.size() - skip_length;
	if (range <= 0) return -1;
	int rnd = random()%range;
	int choice = (stops[mwi]+rnd)%stops.size();
	assert(choice >= stops[mwi] || choice < mwi);
	return choice;
}


struct SentenceMWs  {
	vector <int> stops;
	vector <int> idx;
};
void groupMWs(vector<int>& outIdx, vector<int>& stops, const vector<SeqTok>& s ) {
	vector<pair<int,int> > sorter(s.size());
	outIdx.resize(s.size()); stops.resize(s.size());
	for (size_t i=0; i < s.size(); ++i) sorter[i]=pair<int,int>(s[i].MWhead>=0? s[i].MWhead: i, i);
	sort(sorter.begin(),sorter.end());
	for (size_t i=0; i < s.size(); ++i) outIdx[i] = sorter[i].second;
	for (size_t i=s.size(); i > 0; --i) stops[i-1] = (i < s.size() && sorter[i-1].first == sorter[i].first) ? stops [i] : i;
}

void groupMWs(vector<SentenceMWs> &mws, const vector<vector<SeqTok> > &s ) {
	mws.resize(s.size());
	for (size_t i=0; i<s.size(); ++i)
		groupMWs(mws[i].idx, mws[i].stops, s[i]);
}


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

void miscFeatures(vector<float>& fi, const SeqTok& word, int ncapital) {
	fi.clear();
	bool capital = isCapital(word.word.c_str());
	float capital_norm = capital? 1. / ncapital : 0.;
	float uprRatio = upperNextRatio(word.word.c_str() );
	bool isUrl = !strcasecmp(word.word.c_str(),"URL") || strstr(word.word.c_str(),"://");
	bool isHash = word.word[0]=='#';
	bool isAt = word.word[0]=='@';
	fi.push_back(plusminusbit(capital));
	fi.push_back(plusminusbit(isUrl));
	fi.push_back(plusminusbit(isHash));
	fi.push_back(plusminusbit(isAt));
	fi.push_back(plusminusbit(word.prequoted));
	fi.push_back(plusminusbit(word.postquoted));
	fi.push_back(uprRatio);
	fi.push_back(capital_norm);
	fi.push_back(log(word.lex->range+10)-log(10));
	assert(fi.size()==nMiscFeatures);
}

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

float   learnCompose(const vector<vector<SeqTok> >& s, MWCompModel& mw_model, 	SimpleSGDTrainer& sgd) {
	vector<vector<float> > mean;
	vector<SentenceMWs>  aux;
	calcMean(mean, s);
	groupMWs(aux, s);
	assert (s.size()==mean.size());
	assert (s.size()==aux.size());

		float loss = 0, oneLoss;
	  for (int i=0; i<min(s.size(),5000000UL); ++i) {
			ComputationGraph cg;
			list<vector<float> > dvs;
			Expression m = input(cg, {voc.vsize},&mean[i]);
		for (size_t j=0; j<s[i].size(); j=aux[i].stops[j]) {
			int true_j = aux[i].idx[j];
				POS  p =  s[i][true_j].pos;
				Expression v = input(cg,{voc.vsize},&s[i][true_j].lex->v), h = input(cg,{HASH_DIM},&s[i][true_j].h),
				f = input(cg,{nMiscFeatures},&s[i][true_j].misc);
		for (int n=0; n<1; ++n) {
			//Non Multi word
			int k = chooseRandomOther(aux[i].stops, j) ;
		if (k >= 0) {
			int true_k = aux[i].idx[k];
			  if(!( true_k > true_j && true_k < true_j + 10)) continue;
				dvs.emplace_back();
				vector<float>& dv = dvs.back();
				distFeatures(dv,s[i],true_k,true_j);
				POS pk = s[i][true_k].pos;
				Expression vk = input(cg, {voc.vsize},&s[i][true_k].lex->v), hk = input(cg,{HASH_DIM},&s[i][true_k].h),
					fk = input(cg,{nMiscFeatures},&s[i][true_k].misc),
					comp_v = mw_model.compose_v (v,p,vk,pk),
						comp_h = hash_compose(h,hk),
						d = input(cg,{nDistFeatures},&dv);
			loss+=oneLoss = mw_model. learnOne(v,h,vk,hk,comp_v,comp_h,m,f,d, false);
#ifdef DEBUG_MW_SEL
			cout << "Non-MW##"<<true_j<<", "<<true_k<<" = "<<oneLoss<<"\t";
#endif
			logVectors(vf, 'F', 0, &v,&h,&vk,&hk,&comp_v,&comp_h,&m,&f,&d);
			checkConsistency(i, true_j, true_k, false);
			sgd.update(rate);
		}
	}			
	if (j<s[i].size()-1) for (size_t k=j+1; k<aux[i].stops[j]; ++k) {
			MW++;
			if (k > j+1) longMW++;
			int true_k = aux[i].idx[k];
				//Mutiword! (note: modifies h and v so use it after Non Multi Word)
				dvs.emplace_back();
				vector<float>& dv = dvs.back();
				distFeatures(dv,s[i],aux[i].idx[k-1]/*prev true_k*/,true_k);
				POS pk = s[i][true_k].pos;
				Expression vk = input(cg,{voc.vsize}, &s [i][true_k].lex->v), hk = input(cg,{HASH_DIM},&s[i][true_k].h),
					fk = input(cg,{nMiscFeatures},&s[i][true_k].misc),
					comp_v = mw_model.compose_v  (v,p,vk,pk),
						comp_h = hash_compose(h,hk), comp_f=misc_compose(f,fk),
						d = input(cg,{nDistFeatures},&dv);
				loss += oneLoss = mw_model.learnOne(v,h,vk,hk,comp_v,comp_h,m,f,d,true); 
#ifdef DEBUG_MW_SEL
				cout << "MW##"<<true_j<<", "<<true_k<<" = "<<oneLoss<<"\t";
#endif
				logVectors(vf, 'T', 1, &v,&h,&vk,&hk,&comp_v,&comp_h,&m,&f,&d);
				checkConsistency(i, true_j, true_k, true);
				sgd.update(rate);
				v=comp_v;
				h=comp_h;
				f=comp_f;
				//composition pos is left as in the 1st word. (Voluntaristic)
			}
			
		//Now comp_v and comp_h contain combined senses for Multi word
		loss += oneLoss = mw_model.learnOneSense(v,s[i][true_j].sense);
#ifdef DEBUG_MW_SENSE
		cout << "Learning sense of word #"<<true_j<<", loss = "<<oneLoss<<endl;
#endif

		logVectors(vsf, 'S',  s[i][true_j].sense, &v,&h);
		sgd.update(rate);

}
		cout << "\nSentence " << i<<" , loss = " << loss << endl;
	}
	cerr << "MWs processed: " << MW << ", incl. Long MWs: " << longMW << endl;
	return loss;
}

void calcMisc(vector<vector<SeqTok> > &s ) {
	for (size_t i=0; i<s.size();  i++) {
				  int ncap = std::accumulate(s [i].begin(), s[i].end(), 0, [](int n, const SeqTok& tok) {return isCapital(tok.word.c_str())?n+1:n;});
			for (size_t j=0; j<s[i].size(); ++j) miscFeatures(s[i][j].misc, s[i][j], ncap   );
	}
}

void pExp(Expression * x) {
	vector<float>a (as_vector(x->value() ));
	for (size_t i = 0; i < min(a.size(), 10UL); ++i) cerr << a[i] << " ";
	cerr << endl;
}

void predictMW (vector<vector<SeqTok> >& s, MWCompModel& mw_model) {
	vector<vector<float> > mean;
	calcMean(mean,s);
	for (size_t i=0; i<s.size(); ++i) {

		//clean MW info
					size_t vol = s[i].size();
					bool consumed[vol], inside[vol];
					fill(consumed,	consumed+vol,false);
					fill(inside,	inside+vol,false);
				for (size_t j =0; j<vol; ++j) {
						s[i][j].MWhead  = 0;
						s[i][j].mw =  'O';
					}
		for (int m=0; m<vol; ++m) if (!consumed[m]) {
				cerr<<m<<"\t";
				ComputationGraph cg;
				list<vector<float> > dvs;
				POS p = s[i][m].pos;
				Expression v = input(cg,{voc.vsize},&s[i][m].lex->v), h  = input(cg,{HASH_DIM},&s[i][m].h),
						f = input(cg,{nMiscFeatures},&s[i][m].misc);
			size_t inside_start = m+1;
					Expression mn =  input(cg, {voc.vsize},&mean[i]);

								
			int last_mw_j = m;
			for (int j=m+1; j<vol&&!consumed[j]; ++j) {
				dvs.emplace_back();
				vector<float>& dv = dvs.back();
				distFeatures(dv, s[i], last_mw_j, j);
				POS pj = s[i][j].pos;
				Expression vj = input(cg,{voc.vsize},&s[i][j].lex->v), hj = input(cg,{HASH_DIM},&s[i][j].h),
					fj = input(cg,{nMiscFeatures},&s[i][j].misc),
					comp_v = mw_model.compose_v (v,p,vj,pj),
						comp_h = hash_compose(h,hj),
						comp_f = misc_compose(f,fj),
						d = input(cg,{nDistFeatures},&dv);
				mw_model.classify(v,h,vj,hj,comp_v,comp_h,mn,f,d);
				if (as_scalar(cg.forward()) > 0) {
					cerr<<" MW: "<<j<<"\t";
					consumed [j] = true;
					v = comp_v;
					h = comp_h;
					f = comp_f;
					s[i][m].mw = inside[m]?  'b': 'B';
					s[i][j].MWhead = m+1 ;
					s[i][j].mw = inside[m]?  'i': 'I';
					while(inside_start < j-1) {if (s[i][inside_start].mw=='O') s[i][inside_start].mw= 'o'; inside[inside_start++]=true;}
					last_mw_j = j;
				} else   cerr<<" Non-MW:"<<j<<"\t";
			}
						mw_model.sense(v);
						vector<float> vs(as_vector(cg.forward()));
						s[i][m].sense = max_element(vs.begin(),vs.end())-vs.begin() ;
		}
	}
}

void writeConll(const vector<vector<SeqTok> >& s, MWCompModel::Tables& tables, const string& f) {
	ofstream of(f);
	for (size_t i=0; i<s.size(); ++i) {
		for(size_t j=0; j<s[i].size(); ++j) {
			const SeqTok &tok = s[i][j];
			of << j+1 << "\t" << tok.word << "\t" <<  tok.lemma << "\t" << tables.pos.at(tok.pos) << "\t" <<
				tok.mw << "\t" << tok.MWhead << "\t" << "\t"  << tables.senses.at(tok.sense) << endl;
		}
		of << endl;
	}
  }

void initialize(MWCompModel &model, const string &filename)
{
 cerr << "Initialising model parameters from file: " << filename << endl;
 ifstream in(filename);
 boost::archive::text_iarchive ia(in);
 ia >> model.tables.pos;
 ia >> model.tables.senses;
 model.initialize();
 ia >> model.m;
}

void old_initialize(int posC, int senseC, MWCompModel &model, const string &filename)
{
 cerr << "Initialising model parameters from file: " << filename << endl;
 ifstream in(filename);
 boost::archive::text_iarchive ia(in);
 model.initialize(posC, senseC);
 ia >> model.m;
 model.initialized = false;
 ia >> model.tables.pos;
 ia >> model.tables.senses;
 model.initialized = true;

}

void store(const MWCompModel &model, const string &fname) {
cout << "# saving the model ... "  << flush;
      ofstream out(fname);
      boost::archive::text_oarchive oa(out);
      oa << model.tables.pos;
      oa << model.tables.senses;
      oa << model.m;
      cout << " now  done!" << endl;
}

int main(int argc, char** argv) {
  int opt;
  bool continueLearn(false);
  //temporary for old model loading
  int posCount =-1, senseCount =-1;
  string tFile, lFile, pFile, ptFile, plFile, vFile, mFile;
  while ((opt = getopt (argc, argv, "s:P:v:L:l:T:t:cm:p:")) != -1)
  {
    switch (opt)
    {
      case 'P':
                printf ("Pos count: \"%s\"\n", optarg);
                posCount = strtol(optarg,0,0);
                break;
      case 's':
                printf ("Test file: \"%s\"\n", optarg);
                senseCount = strtol(optarg,0,0);
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
      case 'c':
                printf ("Continuing learn\n");
                continueLearn = true;
                break;
    }
  }
	if (vFile.empty()) {
		cerr << "No vector file specified. Please set (-v filename)" << endl;
		return 1;
	}

  	cnn::Initialize(argc, argv);

	readVectors(voc, vFile.c_str());
	voc.initDerived();
	unknownWord.v.resize( voc.vsize, 0. );
	MWCompModel mw_model;
	if (lFile.empty() || continueLearn) {
		if (mFile.empty()) 
		{
			cerr << "No model specified. Please set either a CoNLL file to learn from (-l filename) or a previously stored model (m filename)" << endl;
			return 1;
		}
			if (posCount >= 0) old_initialize(posCount, senseCount, mw_model, mFile);
			else initialize(mw_model, mFile);
			mw_model.initialized = true;
	}
	if (!lFile.empty()) try {
		vector<vector<SeqTok> > train_s;
		readConll(train_s, mw_model.tables, lFile.c_str());
		if (!plFile.empty()) {
			addParserInfo(train_s,plFile);
		}
		if (!mw_model.initialized) mw_model.initialize();
		calcMisc( train_s )   ;
		SimpleSGDTrainer sgd(&mw_model.m);
		float lastLoss = 1e50;
		for (size_t epoch = 0; epoch < NEPOCH; ++epoch) {
			float loss = learnCompose(train_s, mw_model, sgd);
			if (!mFile.empty() && loss<lastLoss) store(mw_model, mFile);
			lastLoss = loss;
			cout << "Epoch " << epoch<<" , loss = " << loss << endl;
			sgd.update_epoch();
		}
	} catch (exception e) {cerr << e.what() << endl;}

	if (!tFile.empty()) try {
		vector<vector<SeqTok> > test_s;
		readConll(test_s, mw_model.tables, tFile);
		if (ptFile.empty()) {
			addParserInfo(test_s, ptFile);
		}
		calcMisc(test_s);
		predictMW(test_s, mw_model);
		writeConll(test_s, mw_model.tables, pFile);
	} catch (exception e) {{cerr << e.what() << endl;}}

	return 0;
}


