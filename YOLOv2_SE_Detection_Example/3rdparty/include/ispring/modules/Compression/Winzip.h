#if !defined(ISPRING_7E1_9_15_ZIP_HPP_INCLUDED)
#define ISPRING_7E1_9_15_ZIP_HPP_INCLUDED
#ifndef DOXYGEN
#if defined(_WIN32) && defined(__GNUC__)
#if defined(_UNICODE)
#define _tcsncpy_s(a,b,c,d)	wcsncpy(a,c,d)
#define _tcscpy_s(a,b,c)	wcscpy(a,c)
#else
#define _tcsncpy_s(a,b,c,d) strncpy(a,c,d)
#define _tcscpy_s(a,b,c)	strcpy(a,c)
#endif
#define strcpy_s(a,b,c)		strcpy(a,c)
#define wcscpy_s(a,b,c)		wcscpy(a,c)
#define strcat_s(a,b)		strcat(a,b)
#endif
#ifdef _WIN32
#include"../File/FileManager.h"
#include"../Verify/VerifyError.h" 
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include<Windows.h>
#include<vector>
#include<atomic>
#include<climits>

//Unzip part
namespace ispring_3rdparty {
#ifndef _unzip_H
#define _unzip_H

#ifndef _zip_H
	DECLARE_HANDLE(HZIP);
#endif
	typedef DWORD ZRESULT;
	typedef struct {
		int index;                 // index of this file within the zip
		TCHAR name[MAX_PATH];      // filename within the zip
		DWORD attr;                // attributes, as in GetFileAttributes.
		FILETIME atime, ctime, mtime;// access, create, modify filetimes
		long comp_size;            // sizes of item, compressed and uncompressed. These
		long unc_size;             // may be -1 if not yet known (e.g. being streamed in)
	} ZIPENTRY;
	inline HZIP OpenZip(const TCHAR *fn, const char *password);
	inline HZIP OpenZip(void *z, unsigned int len, const char *password);
	inline HZIP OpenZipHandle(HANDLE h, const char *password);
	inline ZRESULT GetZipItem(HZIP hz, int index, ZIPENTRY *ze);
	inline ZRESULT FindZipItem(HZIP hz, const TCHAR *name, bool ic, int *index, ZIPENTRY *ze);
	inline ZRESULT UnzipItem(HZIP hz, int index, const TCHAR *fn);
	inline ZRESULT UnzipItem(HZIP hz, int index, void *z, unsigned int len);
	inline ZRESULT UnzipItemHandle(HZIP hz, int index, HANDLE h);
	inline ZRESULT SetUnzipBaseDir(HZIP hz, const TCHAR *dir);
	inline ZRESULT CloseZip(HZIP hz);
	inline unsigned int FormatZipMessage(ZRESULT code, TCHAR *buf, unsigned int len);

#define ZR_OK         0x00000000     // nb. the pseudo-code zr-recent is never returned,
#define ZR_RECENT     0x00000001     // but can be passed to FormatZipMessage.
	// The following come from general system stuff (e.g. files not openable)
#define ZR_GENMASK    0x0000FF00
#define ZR_NODUPH     0x00000100     // couldn't duplicate the handle
#define ZR_NOFILE     0x00000200     // couldn't create/open the file
#define ZR_NOALLOC    0x00000300     // failed to allocate some resource
#define ZR_WRITE      0x00000400     // a general error writing to the file
#define ZR_NOTFOUND   0x00000500     // couldn't find that file in the zip
#define ZR_MORE       0x00000600     // there's still more data to be unzipped
#define ZR_CORRUPT    0x00000700     // the zipfile is corrupt or not a zipfile
#define ZR_READ       0x00000800     // a general error reading the file
#define ZR_PASSWORD   0x00001000     // we didn't get the right password to unzip the file
	// The following come from mistakes on the part of the caller
#define ZR_CALLERMASK 0x00FF0000
#define ZR_ARGS       0x00010000     // general mistake with the arguments
#define ZR_NOTMMAP    0x00020000     // tried to ZipGetMemory, but that only works on mmap zipfiles, which yours wasn't
#define ZR_MEMSIZE    0x00030000     // the memory size is too small
#define ZR_FAILED     0x00040000     // the thing was already failed when you called this function
#define ZR_ENDED      0x00050000     // the zip creation has already been closed
#define ZR_MISSIZE    0x00060000     // the indicated input file size turned out mistaken
#define ZR_PARTIALUNZ 0x00070000     // the file had already been partially unzipped
#define ZR_ZMODE      0x00080000     // tried to mix creating/opening a zip 
	// The following come from bugs within the zip library itself
#define ZR_BUGMASK    0xFF000000
#define ZR_NOTINITED  0x01000000     // initialisation didn't work
#define ZR_SEEK       0x02000000     // trying to seek in an unseekable file
#define ZR_NOCHANGE   0x04000000     // changed its mind on storage, but not allowed
#define ZR_FLATE      0x05000000     // an internal error in the de/inflation code

	inline ZRESULT CloseZipU(HZIP hz);
	inline unsigned int FormatZipMessageU(ZRESULT code, TCHAR *buf, unsigned int len);
	inline bool IsZipHandleU(HZIP hz);
#ifdef _zip_H
#undef CloseZip
#define CloseZip(hz) (IsZipHandleU(hz)?CloseZipU(hz):CloseZipZ(hz))
#else
#define CloseZip CloseZipU
#define FormatZipMessage FormatZipMessageU
#endif



#endif // _unzip_H


#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tchar.h>

#ifdef UNICODE
#define _tsprintf swprintf
#else
#define _tsprintf sprintf
#endif
#include<sstream>

	// THIS FILE is almost entirely based upon code by Jean-loup Gailly
	// and Mark Adler. It has been modified by Lucian Wischik.
	// The modifications were: incorporate the bugfixes of 1.1.4, allow
	// unzipping to/from handles/pipes/files/memory, encryption, unicode,
	// a windowsish api, and putting everything into a single .cpp file.
	// The original code may be found at http://www.gzip.org/zlib/
	// The original copyright text follows.
	//
	//
	//
	// zlib.h -- interface of the 'zlib' general purpose compression library
	//  version 1.1.3, July 9th, 1998
	//
	//  Copyright (C) 1995-1998 Jean-loup Gailly and Mark Adler
	//
	//  This software is provided 'as-is', without any express or implied
	//  warranty.  In no event will the authors be held liable for any damages
	//  arising from the use of this software.
	//
	//  Permission is granted to anyone to use this software for any purpose,
	//  including commercial applications, and to alter it and redistribute it
	//  freely, subject to the following restrictions:
	//
	//  1. The origin of this software must not be misrepresented; you must not
	//     claim that you wrote the original software. If you use this software
	//     in a product, an acknowledgment in the product documentation would be
	//     appreciated but is not required.
	//  2. Altered source versions must be plainly marked as such, and must not be
	//     misrepresented as being the original software.
	//  3. This notice may not be removed or altered from any source distribution.
	//
	//  Jean-loup Gailly        Mark Adler
	//  jloup@gzip.org          madler@alumni.caltech.edu
	//
	//
	//  The data format used by the zlib library is described by RFCs (Request for
	//  Comments) 1950 to 1952 in the files ftp://ds.internic.net/rfc/rfc1950.txt
	//  (zlib format), rfc1951.txt (deflate format) and rfc1952.txt (gzip format).
	//
	//
	//     The 'zlib' compression library provides in-memory compression and
	//  decompression functions, including integrity checks of the uncompressed
	//  data.  This version of the library supports only one compression method
	//  (deflation) but other algorithms will be added later and will have the same
	//  stream interface.
	//
	//     Compression can be done in a single step if the buffers are large
	//  enough (for example if an input file is mmap'ed), or can be done by
	//  repeated calls of the compression function.  In the latter case, the
	//  application must provide more input and/or consume the output
	//  (providing more output space) before each call.
	//
	//     The library also supports reading and writing files in gzip (.gz) format
	//  with an interface similar to that of stdio.
	//
	//     The library does not install any signal handler. The decoder checks
	//  the consistency of the compressed data, so the library should never
	//  crash even in case of corrupted input.
	//
	// for more info about .ZIP format, see ftp://ftp.cdrom.com/pub/infozip/doc/appnote-970311-iz.zip
	//   PkWare has also a specification at ftp://ftp.pkware.com/probdesc.zip

#define ZIP_HANDLE   1
#define ZIP_FILENAME 2
#define ZIP_MEMORY   3


#define zmalloc(len) malloc(len)

#define zfree(p) free(p)

	typedef unsigned long lutime_t;       // define it ourselves since we don't include time.h

										  /*
										  void *zmalloc(unsigned int len)
										  { char *buf = new char[len+32];
										  for (int i=0; i<16; i++)
										  { buf[i]=i;
										  buf[len+31-i]=i;
										  }
										  *((unsigned int*)buf) = len;
										  char c[1000]; wsprintf(c,"malloc 0x%lx  - %lu",buf+16,len);
										  OutputDebugString(c);
										  return buf+16;
										  }

										  void zfree(void *buf)
										  { char c[1000]; wsprintf(c,"free   0x%lx",buf);
										  OutputDebugString(c);
										  char *p = ((char*)buf)-16;
										  unsigned int len = *((unsigned int*)p);
										  bool blown=false;
										  for (int i=0; i<16; i++)
										  { char lo = p[i];
										  char hi = p[len+31-i];
										  if (hi!=i || (lo!=i && i>4)) blown=true;
										  }
										  if (blown)
										  { OutputDebugString("BLOWN!!!");
										  }
										  delete[] p;
										  }
										  */


	typedef struct tm_unz_s {
		unsigned int tm_sec;            // seconds after the minute - [0,59]
		unsigned int tm_min;            // minutes after the hour - [0,59]
		unsigned int tm_hour;           // hours since midnight - [0,23]
		unsigned int tm_mday;           // day of the month - [1,31]
		unsigned int tm_mon;            // months since January - [0,11]
		unsigned int tm_year;           // years - [1980..2044]
	} tm_unz;



	// ----------------------------------------------------------------------
	// some windows<->linux portability things
	// ----------------------------------------------------------------------
	inline DWORD GetFilePosU(HANDLE hfout) {
		return SetFilePointer(hfout, 0, 0, FILE_CURRENT);
	}

	inline FILETIME timet2filetime(const lutime_t t) {
		LONGLONG i = Int32x32To64(t, 10000000) + 116444736000000000LL;
		FILETIME ft;
		ft.dwLowDateTime = (DWORD)i;
		ft.dwHighDateTime = (DWORD)(i >> 32);
		return ft;
	}

	inline FILETIME dosdatetime2filetime(WORD dosdate, WORD dostime) { // date: bits 0-4 are day of month 1-31. Bits 5-8 are month 1..12. Bits 9-15 are year-1980
																	   // time: bits 0-4 are seconds/2, bits 5-10 are minute 0..59. Bits 11-15 are hour 0..23
		SYSTEMTIME st;
		st.wYear = (WORD)(((dosdate >> 9) & 0x7f) + 1980);
		st.wMonth = (WORD)((dosdate >> 5) & 0xf);
		st.wDay = (WORD)(dosdate & 0x1f);
		st.wHour = (WORD)((dostime >> 11) & 0x1f);
		st.wMinute = (WORD)((dostime >> 5) & 0x3f);
		st.wSecond = (WORD)((dostime & 0x1f) * 2);
		st.wMilliseconds = 0;
		FILETIME ft; SystemTimeToFileTime(&st, &ft);
		return ft;
	}

	inline bool FileExists(const TCHAR *fn) {
		return (GetFileAttributes(fn) != 0xFFFFFFFF);
	}
	// ----------------------------------------------------------------------



	// unz_global_info structure contain global data about the ZIPfile
	typedef struct unz_global_info_s {
		unsigned long number_entry;         // total number of entries in the central dir on this disk
		unsigned long size_comment;         // size of the global comment of the zipfile
	} unz_global_info;

	// unz_file_info contain information about a file in the zipfile
	typedef struct unz_file_info_s {
		unsigned long version;              // version made by                 2 bytes
		unsigned long version_needed;       // version needed to extract       2 bytes
		unsigned long flag;                 // general purpose bit flag        2 bytes
		unsigned long compression_method;   // compression method              2 bytes
		unsigned long dosDate;              // last mod file date in Dos fmt   4 bytes
		unsigned long crc;                  // crc-32                          4 bytes
		unsigned long compressed_size;      // compressed size                 4 bytes
		unsigned long uncompressed_size;    // uncompressed size               4 bytes
		unsigned long size_filename;        // filename length                 2 bytes
		unsigned long size_file_extra;      // extra field length              2 bytes
		unsigned long size_file_comment;    // file comment length             2 bytes
		unsigned long disk_num_start;       // disk number start               2 bytes
		unsigned long internal_fa;          // internal file attributes        2 bytes
		unsigned long external_fa;          // external file attributes        4 bytes
		tm_unz tmu_date;
	} unz_file_info;


#define UNZ_OK                  (0)
#define UNZ_END_OF_LIST_OF_FILE (-100)
#define UNZ_ERRNO               (Z_ERRNO)
#define UNZ_EOF                 (0)
#define UNZ_PARAMERROR          (-102)
#define UNZ_BADZIPFILE          (-103)
#define UNZ_INTERNALERROR       (-104)
#define UNZ_CRCERROR            (-105)
#define UNZ_PASSWORD            (-106)







#define ZLIB_VERSION "1.1.3"


	// Allowed flush values; see deflate() for details
#define Z_NO_FLUSH      0
#define Z_SYNC_FLUSH    2
#define Z_FULL_FLUSH    3
#define Z_FINISH        4


	// compression levels
#define Z_NO_COMPRESSION         0
#define Z_BEST_SPEED             1
#define Z_BEST_COMPRESSION       9
#define Z_DEFAULT_COMPRESSION  (-1)

	// compression strategy; see deflateInit2() for details
#define Z_FILTERED            1
#define Z_HUFFMAN_ONLY        2
#define Z_DEFAULT_STRATEGY    0

	// Possible values of the data_type field
#define Z_BINARY   0
#define Z_ASCII    1
#define Z_UNKNOWN  2

	// The deflate compression method (the only one supported in this version)
#define Z_DEFLATED   8

	// for initializing zalloc, zfree, opaque
#define Z_NULL  0

	// case sensitivity when searching for filenames
#define CASE_SENSITIVE 1
#define CASE_INSENSITIVE 2


	// Return codes for the compression/decompression functions. Negative
	// values are errors, positive values are used for special but normal events.
#define Z_OK            0
#define Z_STREAM_END    1
#define Z_NEED_DICT     2
#define Z_ERRNO        (-1)
#define Z_STREAM_ERROR (-2)
#define Z_DATA_ERROR   (-3)
#define Z_MEM_ERROR    (-4)
#define Z_BUF_ERROR    (-5)
#define Z_VERSION_ERROR (-6)



	// Basic data types
	typedef unsigned char  Byte;  // 8 bits
	typedef unsigned int   uInt;  // 16 bits or more
	typedef unsigned long  uLong; // 32 bits or more
	typedef void *voidpf;
	typedef void     *voidp;
	typedef long z_off_t;












	typedef voidpf(*alloc_func) (voidpf opaque, uInt items, uInt size);
	typedef void(*free_func)  (voidpf opaque, voidpf address);

	struct internal_state;

	typedef struct z_stream_s {
		Byte    *next_in;  // next input byte
		uInt     avail_in;  // number of bytes available at next_in
		uLong    total_in;  // total nb of input bytes read so far

		Byte    *next_out; // next output byte should be put there
		uInt     avail_out; // remaining free space at next_out
		uLong    total_out; // total nb of bytes output so far

		char     *msg;      // last error message, NULL if no error
		struct internal_state *state; // not visible by applications

		alloc_func zalloc;  // used to allocate the internal state
		free_func  zfree;   // used to free the internal state
		voidpf     opaque;  // private data object passed to zalloc and zfree

		int     data_type;  // best guess about the data type: ascii or binary
		uLong   adler;      // adler32 value of the uncompressed data
		uLong   reserved;   // reserved for future use
	} z_stream;

	typedef z_stream *z_streamp;


	//   The application must update next_in and avail_in when avail_in has
	//   dropped to zero. It must update next_out and avail_out when avail_out
	//   has dropped to zero. The application must initialize zalloc, zfree and
	//   opaque before calling the init function. All other fields are set by the
	//   compression library and must not be updated by the application.
	//
	//   The opaque value provided by the application will be passed as the first
	//   parameter for calls of zalloc and zfree. This can be useful for custom
	//   memory management. The compression library attaches no meaning to the
	//   opaque value.
	//
	//   zalloc must return Z_NULL if there is not enough memory for the object.
	//   If zlib is used in a multi-threaded application, zalloc and zfree must be
	//   thread safe.
	//
	//   The fields total_in and total_out can be used for statistics or
	//   progress reports. After compression, total_in holds the total size of
	//   the uncompressed data and may be saved for use in the decompressor
	//   (particularly if the decompressor wants to decompress everything in
	//   a single step).
	//


	// basic functions

	inline const char *zlibVersion();
	// The application can compare zlibVersion and ZLIB_VERSION for consistency.
	// If the first character differs, the library code actually used is
	// not compatible with the zlib.h header file used by the application.
	// This check is automatically made by inflateInit.






	inline int inflate(z_streamp strm, int flush);
	//
	//    inflate decompresses as much data as possible, and stops when the input
	//  buffer becomes empty or the output buffer becomes full. It may some
	//  introduce some output latency (reading input without producing any output)
	//  except when forced to flush.
	//
	//  The detailed semantics are as follows. inflate performs one or both of the
	//  following actions:
	//
	//  - Decompress more input starting at next_in and update next_in and avail_in
	//    accordingly. If not all input can be processed (because there is not
	//    enough room in the output buffer), next_in is updated and processing
	//    will resume at this point for the next call of inflate().
	//
	//  - Provide more output starting at next_out and update next_out and avail_out
	//    accordingly.  inflate() provides as much output as possible, until there
	//    is no more input data or no more space in the output buffer (see below
	//    about the flush parameter).
	//
	//  Before the call of inflate(), the application should ensure that at least
	//  one of the actions is possible, by providing more input and/or consuming
	//  more output, and updating the next_* and avail_* values accordingly.
	//  The application can consume the uncompressed output when it wants, for
	//  example when the output buffer is full (avail_out == 0), or after each
	//  call of inflate(). If inflate returns Z_OK and with zero avail_out, it
	//  must be called again after making room in the output buffer because there
	//  might be more output pending.
	//
	//    If the parameter flush is set to Z_SYNC_FLUSH, inflate flushes as much
	//  output as possible to the output buffer. The flushing behavior of inflate is
	//  not specified for values of the flush parameter other than Z_SYNC_FLUSH
	//  and Z_FINISH, but the current implementation actually flushes as much output
	//  as possible anyway.
	//
	//    inflate() should normally be called until it returns Z_STREAM_END or an
	//  error. However if all decompression is to be performed in a single step
	//  (a single call of inflate), the parameter flush should be set to
	//  Z_FINISH. In this case all pending input is processed and all pending
	//  output is flushed; avail_out must be large enough to hold all the
	//  uncompressed data. (The size of the uncompressed data may have been saved
	//  by the compressor for this purpose.) The next operation on this stream must
	//  be inflateEnd to deallocate the decompression state. The use of Z_FINISH
	//  is never required, but can be used to inform inflate that a faster routine
	//  may be used for the single inflate() call.
	//
	//     If a preset dictionary is needed at this point (see inflateSetDictionary
	//  below), inflate sets strm-adler to the adler32 checksum of the
	//  dictionary chosen by the compressor and returns Z_NEED_DICT; otherwise
	//  it sets strm->adler to the adler32 checksum of all output produced
	//  so far (that is, total_out bytes) and returns Z_OK, Z_STREAM_END or
	//  an error code as described below. At the end of the stream, inflate()
	//  checks that its computed adler32 checksum is equal to that saved by the
	//  compressor and returns Z_STREAM_END only if the checksum is correct.
	//
	//    inflate() returns Z_OK if some progress has been made (more input processed
	//  or more output produced), Z_STREAM_END if the end of the compressed data has
	//  been reached and all uncompressed output has been produced, Z_NEED_DICT if a
	//  preset dictionary is needed at this point, Z_DATA_ERROR if the input data was
	//  corrupted (input stream not conforming to the zlib format or incorrect
	//  adler32 checksum), Z_STREAM_ERROR if the stream structure was inconsistent
	//  (for example if next_in or next_out was NULL), Z_MEM_ERROR if there was not
	//  enough memory, Z_BUF_ERROR if no progress is possible or if there was not
	//  enough room in the output buffer when Z_FINISH is used. In the Z_DATA_ERROR
	//  case, the application may then call inflateSync to look for a good
	//  compression block.
	//


	inline int inflateEnd(z_streamp strm);
	//
	//     All dynamically allocated data structures for this stream are freed.
	//   This function discards any unprocessed input and does not flush any
	//   pending output.
	//
	//     inflateEnd returns Z_OK if success, Z_STREAM_ERROR if the stream state
	//   was inconsistent. In the error case, msg may be set but then points to a
	//   static string (which must not be deallocated).

	// Advanced functions 

	//  The following functions are needed only in some special applications.





	inline int inflateSetDictionary(z_streamp strm,
									const Byte *dictionary,
									uInt  dictLength);
	//
	//     Initializes the decompression dictionary from the given uncompressed byte
	//   sequence. This function must be called immediately after a call of inflate
	//   if this call returned Z_NEED_DICT. The dictionary chosen by the compressor
	//   can be determined from the Adler32 value returned by this call of
	//   inflate. The compressor and decompressor must use exactly the same
	//   dictionary. 
	//
	//     inflateSetDictionary returns Z_OK if success, Z_STREAM_ERROR if a
	//   parameter is invalid (such as NULL dictionary) or the stream state is
	//   inconsistent, Z_DATA_ERROR if the given dictionary doesn't match the
	//   expected one (incorrect Adler32 value). inflateSetDictionary does not
	//   perform any decompression: this will be done by subsequent calls of
	//   inflate().


	inline int inflateSync(z_streamp strm);
	// 
	//    Skips invalid compressed data until a full flush point can be found, or until all
	//  available input is skipped. No output is provided.
	//
	//    inflateSync returns Z_OK if a full flush point has been found, Z_BUF_ERROR
	//  if no more input was provided, Z_DATA_ERROR if no flush point has been found,
	//  or Z_STREAM_ERROR if the stream structure was inconsistent. In the success
	//  case, the application may save the current current value of total_in which
	//  indicates where valid compressed data was found. In the error case, the
	//  application may repeatedly call inflateSync, providing more input each time,
	//  until success or end of the input data.


	inline int inflateReset(z_streamp strm);
	//     This function is equivalent to inflateEnd followed by inflateInit,
	//   but does not free and reallocate all the internal decompression state.
	//   The stream will keep attributes that may have been set by inflateInit2.
	//
	//      inflateReset returns Z_OK if success, or Z_STREAM_ERROR if the source
	//   stream state was inconsistent (such as zalloc or state being NULL).
	//



	// checksum functions
	// These functions are not related to compression but are exported
	// anyway because they might be useful in applications using the
	// compression library.

	inline uLong adler32(uLong adler, const Byte *buf, uInt len);
	//     Update a running Adler-32 checksum with the bytes buf[0..len-1] and
	//   return the updated checksum. If buf is NULL, this function returns
	//   the required initial value for the checksum.
	//   An Adler-32 checksum is almost as reliable as a CRC32 but can be computed
	//   much faster. Usage example:
	//
	//     uLong adler = adler32(0L, Z_NULL, 0);
	//
	//     while (read_buffer(buffer, length) != EOF) {
	//       adler = adler32(adler, buffer, length);
	//     }
	//     if (adler != original_adler) error();

	inline uLong ucrc32(uLong crc, const Byte *buf, uInt len);
	//     Update a running crc with the bytes buf[0..len-1] and return the updated
	//   crc. If buf is NULL, this function returns the required initial value
	//   for the crc. Pre- and post-conditioning (one's complement) is performed
	//   within this function so it shouldn't be done by the application.
	//   Usage example:
	//
	//     uLong crc = crc32(0L, Z_NULL, 0);
	//
	//     while (read_buffer(buffer, length) != EOF) {
	//       crc = crc32(crc, buffer, length);
	//     }
	//     if (crc != original_crc) error();




	inline const char   *zError(int err);
	inline int           inflateSyncPoint(z_streamp z);
	inline const uLong *get_crc_table(void);



	typedef unsigned char  uch;
	typedef uch uchf;
	typedef unsigned short ush;
	typedef ush ushf;
	typedef unsigned long  ulg;



	const char * const z_errmsg[10] = { // indexed by 2-zlib_error
		"need dictionary",     // Z_NEED_DICT       2
		"stream end",          // Z_STREAM_END      1
		"",                    // Z_OK              0
		"file error",          // Z_ERRNO         (-1)
		"stream error",        // Z_STREAM_ERROR  (-2)
		"data error",          // Z_DATA_ERROR    (-3)
		"insufficient memory", // Z_MEM_ERROR     (-4)
		"buffer error",        // Z_BUF_ERROR     (-5)
		"incompatible version",// Z_VERSION_ERROR (-6)
		"" };


#define ERR_MSG(err) z_errmsg[Z_NEED_DICT-(err)]

#define ERR_RETURN(strm,err) \
  return (strm->msg = (char*)ERR_MSG(err), (err))
	// To be used only when the state is known to be valid 

	// common constants


#define STORED_BLOCK 0
#define STATIC_TREES 1
#define DYN_TREES    2
	// The three kinds of block type 

#define MIN_MATCH  3
#define MAX_MATCH  258
	// The minimum and maximum match lengths 

#define PRESET_DICT 0x20 // preset dictionary flag in zlib header 

	// target dependencies 

#define OS_CODE  0x0b  // Window 95 & Windows NT



	// functions 

#define zmemzero(dest, len) memset(dest, 0, len)

	// Diagnostic functions
#define LuAssert(cond,msg)
#define LuTrace(x)
#define LuTracev(x)
#define LuTracevv(x)
#define LuTracec(c,x)
#define LuTracecv(c,x)


	typedef uLong(*check_func) (uLong check, const Byte *buf, uInt len);
	inline voidpf zcalloc(voidpf opaque, unsigned items, unsigned size);
	inline void   zcfree(voidpf opaque, voidpf ptr);

#define ZALLOC(strm, items, size) \
           (*((strm)->zalloc))((strm)->opaque, (items), (size))
#define ZFREE(strm, addr)  (*((strm)->zfree))((strm)->opaque, (voidpf)(addr))

	//void ZFREE(z_streamp strm,voidpf addr)
	//{ *((strm)->zfree))((strm)->opaque, addr);
	//}

#define TRY_FREE(s, p) {if (p) ZFREE(s, p);}




	// Huffman code lookup table entry--this entry is four bytes for machines
	// that have 16-bit pointers (e.g. PC's in the small or medium model).


	typedef struct inflate_huft_s inflate_huft;

	struct inflate_huft_s {
		union {
			struct {
				Byte Exop;        // number of extra bits or operation
				Byte Bits;        // number of bits in this code or subcode
			} what;
			uInt pad;           // pad structure to a power of 2 (4 bytes for
		} word;               //  16-bit, 8 bytes for 32-bit int's)
		uInt base;            // literal, length base, distance base, or table offset
	};

	// Maximum size of dynamic tree.  The maximum found in a long but non-
	//   exhaustive search was 1004 huft structures (850 for length/literals
	//   and 154 for distances, the latter actually the result of an
	//   exhaustive search).  The actual maximum is not known, but the
	//   value below is more than safe.
#define MANY 1440

	inline int inflate_trees_bits(
		uInt *,                    // 19 code lengths
		uInt *,                    // bits tree desired/actual depth
		inflate_huft * *,       // bits tree result
		inflate_huft *,             // space for trees
		z_streamp);                // for messages

	inline int inflate_trees_dynamic(
		uInt,                       // number of literal/length codes
		uInt,                       // number of distance codes
		uInt *,                    // that many (total) code lengths
		uInt *,                    // literal desired/actual bit depth
		uInt *,                    // distance desired/actual bit depth
		inflate_huft * *,       // literal/length tree result
		inflate_huft * *,       // distance tree result
		inflate_huft *,             // space for trees
		z_streamp);                // for messages

	inline int inflate_trees_fixed(
		uInt *,                    // literal desired/actual bit depth
		uInt *,                    // distance desired/actual bit depth
		const inflate_huft * *,       // literal/length tree result
		const inflate_huft * *,       // distance tree result
		z_streamp);                // for memory allocation





	struct inflate_blocks_state;
	typedef struct inflate_blocks_state inflate_blocks_statef;

	inline inflate_blocks_statef * inflate_blocks_new(
		z_streamp z,
		check_func c,               // check function
		uInt w);                   // window size

	inline int inflate_blocks(
		inflate_blocks_statef *,
		z_streamp,
		int);                      // initial return code

	inline void inflate_blocks_reset(
		inflate_blocks_statef *,
		z_streamp,
		uLong *);                  // check value on output

	inline int inflate_blocks_free(
		inflate_blocks_statef *,
		z_streamp);

	inline void inflate_set_dictionary(
		inflate_blocks_statef *s,
		const Byte *d,  // dictionary
		uInt  n);       // dictionary length

	inline int inflate_blocks_sync_point(
		inflate_blocks_statef *s);




	struct inflate_codes_state;
	typedef struct inflate_codes_state inflate_codes_statef;

	inline inflate_codes_statef *inflate_codes_new(
		uInt, uInt,
		const inflate_huft *, const inflate_huft *,
		z_streamp);

	inline int inflate_codes(
		inflate_blocks_statef *,
		z_streamp,
		int);

	inline void inflate_codes_free(
		inflate_codes_statef *,
		z_streamp);




	typedef enum {
		IBM_TYPE,     // get type bits (3, including end bit)
		IBM_LENS,     // get lengths for stored
		IBM_STORED,   // processing stored block
		IBM_TABLE,    // get table lengths
		IBM_BTREE,    // get bit lengths tree for a dynamic block
		IBM_DTREE,    // get length, distance trees for a dynamic block
		IBM_CODES,    // processing fixed or dynamic block
		IBM_DRY,      // output remaining window bytes
		IBM_DONE,     // finished last block, done 
		IBM_BAD
	}      // got a data error--stuck here 
	inflate_block_mode;

	// inflate blocks semi-private state 
	struct inflate_blocks_state {

		// mode 
		inflate_block_mode  mode;     // current inflate_block mode 

									  // mode dependent information 
		union {
			uInt left;          // if STORED, bytes left to copy 
			struct {
				uInt table;               // table lengths (14 bits) 
				uInt index;               // index into blens (or border)
				uInt *blens;             // bit lengths of codes
				uInt bb;                  // bit length tree depth 
				inflate_huft *tb;         // bit length decoding tree 
			} trees;            // if DTREE, decoding info for trees 
			struct {
				inflate_codes_statef
					*codes;
			} decode;           // if CODES, current state 
		} sub;                // submode
		uInt last;            // true if this block is the last block 

							  // mode independent information 
		uInt bitk;            // bits in bit buffer 
		uLong bitb;           // bit buffer 
		inflate_huft *hufts;  // single malloc for tree space 
		Byte *window;        // sliding window 
		Byte *end;           // one byte after sliding window 
		Byte *read;          // window read pointer 
		Byte *write;         // window write pointer 
		check_func checkfn;   // check function 
		uLong check;          // check on output 

	};


	// defines for inflate input/output
	//   update pointers and return 
#define UPDBITS {s->bitb=b;s->bitk=k;}
#define UPDIN {z->avail_in=n;z->total_in+=(uLong)(p-z->next_in);z->next_in=p;}
#define UPDOUT {s->write=q;}
#define UPDATE {UPDBITS UPDIN UPDOUT}
#define LEAVE {UPDATE return inflate_flush(s,z,r);}
	//   get bytes and bits 
#define LOADIN {p=z->next_in;n=z->avail_in;b=s->bitb;k=s->bitk;}
#define NEEDBYTE {if(n)r=Z_OK;else LEAVE}
#define NEXTBYTE (n--,*p++)
#define NEEDBITS(j) {while(k<(j)){NEEDBYTE;b|=((uLong)NEXTBYTE)<<k;k+=8;}}
#define DUMPBITS(j) {b>>=(j);k-=(j);}
	//   output bytes 
#define WAVAIL (uInt)(q<s->read?s->read-q-1:s->end-q)
#define LOADOUT {q=s->write;m=(uInt)WAVAIL;m;}
#define WRAP {if(q==s->end&&s->read!=s->window){q=s->window;m=(uInt)WAVAIL;}}
#define FLUSH {UPDOUT r=inflate_flush(s,z,r); LOADOUT}
#define NEEDOUT {if(m==0){WRAP if(m==0){FLUSH WRAP if(m==0) LEAVE}}r=Z_OK;}
#define OUTBYTE(a) {*q++=(Byte)(a);m--;}
	//   load local pointers 
#define LOAD {LOADIN LOADOUT}

	// masks for lower bits (size given to avoid silly warnings with Visual C++) 
	// And'ing with mask[n] masks the lower n bits
	const uInt inflate_mask[17] = {
		0x0000,
		0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff,
		0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff
	};

	// copy as much as possible from the sliding window to the output area
	inline int inflate_flush(inflate_blocks_statef *, z_streamp, int);

	inline int inflate_fast(uInt, uInt, const inflate_huft *, const inflate_huft *, inflate_blocks_statef *, z_streamp);



	const uInt fixed_bl = 9;
	const uInt fixed_bd = 5;
	const inflate_huft fixed_tl[] = {
		{ { { 96,7 } },256 },{ { { 0,8 } },80 },{ { { 0,8 } },16 },{ { { 84,8 } },115 },
		{ { { 82,7 } },31 },{ { { 0,8 } },112 },{ { { 0,8 } },48 },{ { { 0,9 } },192 },
		{ { { 80,7 } },10 },{ { { 0,8 } },96 },{ { { 0,8 } },32 },{ { { 0,9 } },160 },
		{ { { 0,8 } },0 },{ { { 0,8 } },128 },{ { { 0,8 } },64 },{ { { 0,9 } },224 },
		{ { { 80,7 } },6 },{ { { 0,8 } },88 },{ { { 0,8 } },24 },{ { { 0,9 } },144 },
		{ { { 83,7 } },59 },{ { { 0,8 } },120 },{ { { 0,8 } },56 },{ { { 0,9 } },208 },
		{ { { 81,7 } },17 },{ { { 0,8 } },104 },{ { { 0,8 } },40 },{ { { 0,9 } },176 },
		{ { { 0,8 } },8 },{ { { 0,8 } },136 },{ { { 0,8 } },72 },{ { { 0,9 } },240 },
		{ { { 80,7 } },4 },{ { { 0,8 } },84 },{ { { 0,8 } },20 },{ { { 85,8 } },227 },
		{ { { 83,7 } },43 },{ { { 0,8 } },116 },{ { { 0,8 } },52 },{ { { 0,9 } },200 },
		{ { { 81,7 } },13 },{ { { 0,8 } },100 },{ { { 0,8 } },36 },{ { { 0,9 } },168 },
		{ { { 0,8 } },4 },{ { { 0,8 } },132 },{ { { 0,8 } },68 },{ { { 0,9 } },232 },
		{ { { 80,7 } },8 },{ { { 0,8 } },92 },{ { { 0,8 } },28 },{ { { 0,9 } },152 },
		{ { { 84,7 } },83 },{ { { 0,8 } },124 },{ { { 0,8 } },60 },{ { { 0,9 } },216 },
		{ { { 82,7 } },23 },{ { { 0,8 } },108 },{ { { 0,8 } },44 },{ { { 0,9 } },184 },
		{ { { 0,8 } },12 },{ { { 0,8 } },140 },{ { { 0,8 } },76 },{ { { 0,9 } },248 },
		{ { { 80,7 } },3 },{ { { 0,8 } },82 },{ { { 0,8 } },18 },{ { { 85,8 } },163 },
		{ { { 83,7 } },35 },{ { { 0,8 } },114 },{ { { 0,8 } },50 },{ { { 0,9 } },196 },
		{ { { 81,7 } },11 },{ { { 0,8 } },98 },{ { { 0,8 } },34 },{ { { 0,9 } },164 },
		{ { { 0,8 } },2 },{ { { 0,8 } },130 },{ { { 0,8 } },66 },{ { { 0,9 } },228 },
		{ { { 80,7 } },7 },{ { { 0,8 } },90 },{ { { 0,8 } },26 },{ { { 0,9 } },148 },
		{ { { 84,7 } },67 },{ { { 0,8 } },122 },{ { { 0,8 } },58 },{ { { 0,9 } },212 },
		{ { { 82,7 } },19 },{ { { 0,8 } },106 },{ { { 0,8 } },42 },{ { { 0,9 } },180 },
		{ { { 0,8 } },10 },{ { { 0,8 } },138 },{ { { 0,8 } },74 },{ { { 0,9 } },244 },
		{ { { 80,7 } },5 },{ { { 0,8 } },86 },{ { { 0,8 } },22 },{ { { 192,8 } },0 },
		{ { { 83,7 } },51 },{ { { 0,8 } },118 },{ { { 0,8 } },54 },{ { { 0,9 } },204 },
		{ { { 81,7 } },15 },{ { { 0,8 } },102 },{ { { 0,8 } },38 },{ { { 0,9 } },172 },
		{ { { 0,8 } },6 },{ { { 0,8 } },134 },{ { { 0,8 } },70 },{ { { 0,9 } },236 },
		{ { { 80,7 } },9 },{ { { 0,8 } },94 },{ { { 0,8 } },30 },{ { { 0,9 } },156 },
		{ { { 84,7 } },99 },{ { { 0,8 } },126 },{ { { 0,8 } },62 },{ { { 0,9 } },220 },
		{ { { 82,7 } },27 },{ { { 0,8 } },110 },{ { { 0,8 } },46 },{ { { 0,9 } },188 },
		{ { { 0,8 } },14 },{ { { 0,8 } },142 },{ { { 0,8 } },78 },{ { { 0,9 } },252 },
		{ { { 96,7 } },256 },{ { { 0,8 } },81 },{ { { 0,8 } },17 },{ { { 85,8 } },131 },
		{ { { 82,7 } },31 },{ { { 0,8 } },113 },{ { { 0,8 } },49 },{ { { 0,9 } },194 },
		{ { { 80,7 } },10 },{ { { 0,8 } },97 },{ { { 0,8 } },33 },{ { { 0,9 } },162 },
		{ { { 0,8 } },1 },{ { { 0,8 } },129 },{ { { 0,8 } },65 },{ { { 0,9 } },226 },
		{ { { 80,7 } },6 },{ { { 0,8 } },89 },{ { { 0,8 } },25 },{ { { 0,9 } },146 },
		{ { { 83,7 } },59 },{ { { 0,8 } },121 },{ { { 0,8 } },57 },{ { { 0,9 } },210 },
		{ { { 81,7 } },17 },{ { { 0,8 } },105 },{ { { 0,8 } },41 },{ { { 0,9 } },178 },
		{ { { 0,8 } },9 },{ { { 0,8 } },137 },{ { { 0,8 } },73 },{ { { 0,9 } },242 },
		{ { { 80,7 } },4 },{ { { 0,8 } },85 },{ { { 0,8 } },21 },{ { { 80,8 } },258 },
		{ { { 83,7 } },43 },{ { { 0,8 } },117 },{ { { 0,8 } },53 },{ { { 0,9 } },202 },
		{ { { 81,7 } },13 },{ { { 0,8 } },101 },{ { { 0,8 } },37 },{ { { 0,9 } },170 },
		{ { { 0,8 } },5 },{ { { 0,8 } },133 },{ { { 0,8 } },69 },{ { { 0,9 } },234 },
		{ { { 80,7 } },8 },{ { { 0,8 } },93 },{ { { 0,8 } },29 },{ { { 0,9 } },154 },
		{ { { 84,7 } },83 },{ { { 0,8 } },125 },{ { { 0,8 } },61 },{ { { 0,9 } },218 },
		{ { { 82,7 } },23 },{ { { 0,8 } },109 },{ { { 0,8 } },45 },{ { { 0,9 } },186 },
		{ { { 0,8 } },13 },{ { { 0,8 } },141 },{ { { 0,8 } },77 },{ { { 0,9 } },250 },
		{ { { 80,7 } },3 },{ { { 0,8 } },83 },{ { { 0,8 } },19 },{ { { 85,8 } },195 },
		{ { { 83,7 } },35 },{ { { 0,8 } },115 },{ { { 0,8 } },51 },{ { { 0,9 } },198 },
		{ { { 81,7 } },11 },{ { { 0,8 } },99 },{ { { 0,8 } },35 },{ { { 0,9 } },166 },
		{ { { 0,8 } },3 },{ { { 0,8 } },131 },{ { { 0,8 } },67 },{ { { 0,9 } },230 },
		{ { { 80,7 } },7 },{ { { 0,8 } },91 },{ { { 0,8 } },27 },{ { { 0,9 } },150 },
		{ { { 84,7 } },67 },{ { { 0,8 } },123 },{ { { 0,8 } },59 },{ { { 0,9 } },214 },
		{ { { 82,7 } },19 },{ { { 0,8 } },107 },{ { { 0,8 } },43 },{ { { 0,9 } },182 },
		{ { { 0,8 } },11 },{ { { 0,8 } },139 },{ { { 0,8 } },75 },{ { { 0,9 } },246 },
		{ { { 80,7 } },5 },{ { { 0,8 } },87 },{ { { 0,8 } },23 },{ { { 192,8 } },0 },
		{ { { 83,7 } },51 },{ { { 0,8 } },119 },{ { { 0,8 } },55 },{ { { 0,9 } },206 },
		{ { { 81,7 } },15 },{ { { 0,8 } },103 },{ { { 0,8 } },39 },{ { { 0,9 } },174 },
		{ { { 0,8 } },7 },{ { { 0,8 } },135 },{ { { 0,8 } },71 },{ { { 0,9 } },238 },
		{ { { 80,7 } },9 },{ { { 0,8 } },95 },{ { { 0,8 } },31 },{ { { 0,9 } },158 },
		{ { { 84,7 } },99 },{ { { 0,8 } },127 },{ { { 0,8 } },63 },{ { { 0,9 } },222 },
		{ { { 82,7 } },27 },{ { { 0,8 } },111 },{ { { 0,8 } },47 },{ { { 0,9 } },190 },
		{ { { 0,8 } },15 },{ { { 0,8 } },143 },{ { { 0,8 } },79 },{ { { 0,9 } },254 },
		{ { { 96,7 } },256 },{ { { 0,8 } },80 },{ { { 0,8 } },16 },{ { { 84,8 } },115 },
		{ { { 82,7 } },31 },{ { { 0,8 } },112 },{ { { 0,8 } },48 },{ { { 0,9 } },193 },
		{ { { 80,7 } },10 },{ { { 0,8 } },96 },{ { { 0,8 } },32 },{ { { 0,9 } },161 },
		{ { { 0,8 } },0 },{ { { 0,8 } },128 },{ { { 0,8 } },64 },{ { { 0,9 } },225 },
		{ { { 80,7 } },6 },{ { { 0,8 } },88 },{ { { 0,8 } },24 },{ { { 0,9 } },145 },
		{ { { 83,7 } },59 },{ { { 0,8 } },120 },{ { { 0,8 } },56 },{ { { 0,9 } },209 },
		{ { { 81,7 } },17 },{ { { 0,8 } },104 },{ { { 0,8 } },40 },{ { { 0,9 } },177 },
		{ { { 0,8 } },8 },{ { { 0,8 } },136 },{ { { 0,8 } },72 },{ { { 0,9 } },241 },
		{ { { 80,7 } },4 },{ { { 0,8 } },84 },{ { { 0,8 } },20 },{ { { 85,8 } },227 },
		{ { { 83,7 } },43 },{ { { 0,8 } },116 },{ { { 0,8 } },52 },{ { { 0,9 } },201 },
		{ { { 81,7 } },13 },{ { { 0,8 } },100 },{ { { 0,8 } },36 },{ { { 0,9 } },169 },
		{ { { 0,8 } },4 },{ { { 0,8 } },132 },{ { { 0,8 } },68 },{ { { 0,9 } },233 },
		{ { { 80,7 } },8 },{ { { 0,8 } },92 },{ { { 0,8 } },28 },{ { { 0,9 } },153 },
		{ { { 84,7 } },83 },{ { { 0,8 } },124 },{ { { 0,8 } },60 },{ { { 0,9 } },217 },
		{ { { 82,7 } },23 },{ { { 0,8 } },108 },{ { { 0,8 } },44 },{ { { 0,9 } },185 },
		{ { { 0,8 } },12 },{ { { 0,8 } },140 },{ { { 0,8 } },76 },{ { { 0,9 } },249 },
		{ { { 80,7 } },3 },{ { { 0,8 } },82 },{ { { 0,8 } },18 },{ { { 85,8 } },163 },
		{ { { 83,7 } },35 },{ { { 0,8 } },114 },{ { { 0,8 } },50 },{ { { 0,9 } },197 },
		{ { { 81,7 } },11 },{ { { 0,8 } },98 },{ { { 0,8 } },34 },{ { { 0,9 } },165 },
		{ { { 0,8 } },2 },{ { { 0,8 } },130 },{ { { 0,8 } },66 },{ { { 0,9 } },229 },
		{ { { 80,7 } },7 },{ { { 0,8 } },90 },{ { { 0,8 } },26 },{ { { 0,9 } },149 },
		{ { { 84,7 } },67 },{ { { 0,8 } },122 },{ { { 0,8 } },58 },{ { { 0,9 } },213 },
		{ { { 82,7 } },19 },{ { { 0,8 } },106 },{ { { 0,8 } },42 },{ { { 0,9 } },181 },
		{ { { 0,8 } },10 },{ { { 0,8 } },138 },{ { { 0,8 } },74 },{ { { 0,9 } },245 },
		{ { { 80,7 } },5 },{ { { 0,8 } },86 },{ { { 0,8 } },22 },{ { { 192,8 } },0 },
		{ { { 83,7 } },51 },{ { { 0,8 } },118 },{ { { 0,8 } },54 },{ { { 0,9 } },205 },
		{ { { 81,7 } },15 },{ { { 0,8 } },102 },{ { { 0,8 } },38 },{ { { 0,9 } },173 },
		{ { { 0,8 } },6 },{ { { 0,8 } },134 },{ { { 0,8 } },70 },{ { { 0,9 } },237 },
		{ { { 80,7 } },9 },{ { { 0,8 } },94 },{ { { 0,8 } },30 },{ { { 0,9 } },157 },
		{ { { 84,7 } },99 },{ { { 0,8 } },126 },{ { { 0,8 } },62 },{ { { 0,9 } },221 },
		{ { { 82,7 } },27 },{ { { 0,8 } },110 },{ { { 0,8 } },46 },{ { { 0,9 } },189 },
		{ { { 0,8 } },14 },{ { { 0,8 } },142 },{ { { 0,8 } },78 },{ { { 0,9 } },253 },
		{ { { 96,7 } },256 },{ { { 0,8 } },81 },{ { { 0,8 } },17 },{ { { 85,8 } },131 },
		{ { { 82,7 } },31 },{ { { 0,8 } },113 },{ { { 0,8 } },49 },{ { { 0,9 } },195 },
		{ { { 80,7 } },10 },{ { { 0,8 } },97 },{ { { 0,8 } },33 },{ { { 0,9 } },163 },
		{ { { 0,8 } },1 },{ { { 0,8 } },129 },{ { { 0,8 } },65 },{ { { 0,9 } },227 },
		{ { { 80,7 } },6 },{ { { 0,8 } },89 },{ { { 0,8 } },25 },{ { { 0,9 } },147 },
		{ { { 83,7 } },59 },{ { { 0,8 } },121 },{ { { 0,8 } },57 },{ { { 0,9 } },211 },
		{ { { 81,7 } },17 },{ { { 0,8 } },105 },{ { { 0,8 } },41 },{ { { 0,9 } },179 },
		{ { { 0,8 } },9 },{ { { 0,8 } },137 },{ { { 0,8 } },73 },{ { { 0,9 } },243 },
		{ { { 80,7 } },4 },{ { { 0,8 } },85 },{ { { 0,8 } },21 },{ { { 80,8 } },258 },
		{ { { 83,7 } },43 },{ { { 0,8 } },117 },{ { { 0,8 } },53 },{ { { 0,9 } },203 },
		{ { { 81,7 } },13 },{ { { 0,8 } },101 },{ { { 0,8 } },37 },{ { { 0,9 } },171 },
		{ { { 0,8 } },5 },{ { { 0,8 } },133 },{ { { 0,8 } },69 },{ { { 0,9 } },235 },
		{ { { 80,7 } },8 },{ { { 0,8 } },93 },{ { { 0,8 } },29 },{ { { 0,9 } },155 },
		{ { { 84,7 } },83 },{ { { 0,8 } },125 },{ { { 0,8 } },61 },{ { { 0,9 } },219 },
		{ { { 82,7 } },23 },{ { { 0,8 } },109 },{ { { 0,8 } },45 },{ { { 0,9 } },187 },
		{ { { 0,8 } },13 },{ { { 0,8 } },141 },{ { { 0,8 } },77 },{ { { 0,9 } },251 },
		{ { { 80,7 } },3 },{ { { 0,8 } },83 },{ { { 0,8 } },19 },{ { { 85,8 } },195 },
		{ { { 83,7 } },35 },{ { { 0,8 } },115 },{ { { 0,8 } },51 },{ { { 0,9 } },199 },
		{ { { 81,7 } },11 },{ { { 0,8 } },99 },{ { { 0,8 } },35 },{ { { 0,9 } },167 },
		{ { { 0,8 } },3 },{ { { 0,8 } },131 },{ { { 0,8 } },67 },{ { { 0,9 } },231 },
		{ { { 80,7 } },7 },{ { { 0,8 } },91 },{ { { 0,8 } },27 },{ { { 0,9 } },151 },
		{ { { 84,7 } },67 },{ { { 0,8 } },123 },{ { { 0,8 } },59 },{ { { 0,9 } },215 },
		{ { { 82,7 } },19 },{ { { 0,8 } },107 },{ { { 0,8 } },43 },{ { { 0,9 } },183 },
		{ { { 0,8 } },11 },{ { { 0,8 } },139 },{ { { 0,8 } },75 },{ { { 0,9 } },247 },
		{ { { 80,7 } },5 },{ { { 0,8 } },87 },{ { { 0,8 } },23 },{ { { 192,8 } },0 },
		{ { { 83,7 } },51 },{ { { 0,8 } },119 },{ { { 0,8 } },55 },{ { { 0,9 } },207 },
		{ { { 81,7 } },15 },{ { { 0,8 } },103 },{ { { 0,8 } },39 },{ { { 0,9 } },175 },
		{ { { 0,8 } },7 },{ { { 0,8 } },135 },{ { { 0,8 } },71 },{ { { 0,9 } },239 },
		{ { { 80,7 } },9 },{ { { 0,8 } },95 },{ { { 0,8 } },31 },{ { { 0,9 } },159 },
		{ { { 84,7 } },99 },{ { { 0,8 } },127 },{ { { 0,8 } },63 },{ { { 0,9 } },223 },
		{ { { 82,7 } },27 },{ { { 0,8 } },111 },{ { { 0,8 } },47 },{ { { 0,9 } },191 },
		{ { { 0,8 } },15 },{ { { 0,8 } },143 },{ { { 0,8 } },79 },{ { { 0,9 } },255 }
	};
	const inflate_huft fixed_td[] = {
		{ { { 80,5 } },1 },{ { { 87,5 } },257 },{ { { 83,5 } },17 },{ { { 91,5 } },4097 },
		{ { { 81,5 } },5 },{ { { 89,5 } },1025 },{ { { 85,5 } },65 },{ { { 93,5 } },16385 },
		{ { { 80,5 } },3 },{ { { 88,5 } },513 },{ { { 84,5 } },33 },{ { { 92,5 } },8193 },
		{ { { 82,5 } },9 },{ { { 90,5 } },2049 },{ { { 86,5 } },129 },{ { { 192,5 } },24577 },
		{ { { 80,5 } },2 },{ { { 87,5 } },385 },{ { { 83,5 } },25 },{ { { 91,5 } },6145 },
		{ { { 81,5 } },7 },{ { { 89,5 } },1537 },{ { { 85,5 } },97 },{ { { 93,5 } },24577 },
		{ { { 80,5 } },4 },{ { { 88,5 } },769 },{ { { 84,5 } },49 },{ { { 92,5 } },12289 },
		{ { { 82,5 } },13 },{ { { 90,5 } },3073 },{ { { 86,5 } },193 },{ { { 192,5 } },24577 }
	};







	// copy as much as possible from the sliding window to the output area
	inline int inflate_flush(inflate_blocks_statef *s, z_streamp z, int r) {
		uInt n;
		Byte *p;
		Byte *q;

		// local copies of source and destination pointers 
		p = z->next_out;
		q = s->read;

		// compute number of bytes to copy as far as end of window 
		n = (uInt)((q <= s->write ? s->write : s->end) - q);
		if (n > z->avail_out) n = z->avail_out;
		if (n && r == Z_BUF_ERROR) r = Z_OK;

		// update counters
		z->avail_out -= n;
		z->total_out += n;

		// update check information 
		if (s->checkfn != Z_NULL)
			z->adler = s->check = (*s->checkfn)(s->check, q, n);

		// copy as far as end of window 
		if (n != 0)          // check for n!=0 to avoid waking up CodeGuard
		{
			memcpy(p, q, n);
			p += n;
			q += n;
		}

		// see if more to copy at beginning of window
		if (q == s->end) {
			// wrap pointers 
			q = s->window;
			if (s->write == s->end)
				s->write = s->window;

			// compute bytes to copy 
			n = (uInt)(s->write - q);
			if (n > z->avail_out) n = z->avail_out;
			if (n && r == Z_BUF_ERROR) r = Z_OK;

			// update counters 
			z->avail_out -= n;
			z->total_out += n;

			// update check information 
			if (s->checkfn != Z_NULL)
				z->adler = s->check = (*s->checkfn)(s->check, q, n);

			// copy
			if (n != 0) {
				memcpy(p, q, n); p += n; q += n;
			}
		}

		// update pointers
		z->next_out = p;
		s->read = q;

		// done
		return r;
	}






	// simplify the use of the inflate_huft type with some defines
#define exop word.what.Exop
#define Z_bits word.what.Bits

	typedef enum {        // waiting for "i:"=input, "o:"=output, "x:"=nothing 
		START,    // x: set up for LEN 
		LEN,      // i: get length/literal/eob next 
		LENEXT,   // i: getting length extra (have base) 
		DIST,     // i: get distance next 
		DISTEXT,  // i: getting distance extra 
		COPY,     // o: copying bytes in window, waiting for space
		LIT,      // o: got literal, waiting for output space 
		WASH,     // o: got eob, possibly still output waiting 
		END,      // x: got eob and all data flushed 
		BADCODE
	}  // x: got error 
	inflate_codes_mode;

	// inflate codes private state
	struct inflate_codes_state {

		// mode 
		inflate_codes_mode mode;      // current inflate_codes mode 

									  // mode dependent information 
		uInt len;
		union {
			struct {
				const inflate_huft *tree;       // pointer into tree 
				uInt need;                // bits needed 
			} code;             // if LEN or DIST, where in tree 
			uInt lit;           // if LIT, literal 
			struct {
				uInt get;                 // bits to get for extra 
				uInt dist;                // distance back to copy from 
			} copy;             // if EXT or COPY, where and how much 
		} sub;                // submode

							  // mode independent information 
		Byte lbits;           // ltree bits decoded per branch 
		Byte dbits;           // dtree bits decoder per branch 
		const inflate_huft *ltree;          // literal/length/eob tree
		const inflate_huft *dtree;          // distance tree

	};


	inline inflate_codes_statef *inflate_codes_new(
		uInt bl, uInt bd,
		const inflate_huft *tl,
		const inflate_huft *td, // need separate declaration for Borland C++
		z_streamp z) {
		inflate_codes_statef *c;

		if ((c = (inflate_codes_statef *)
			 ZALLOC(z, 1, sizeof(struct inflate_codes_state))) != Z_NULL) {
			c->mode = START;
			c->lbits = (Byte)bl;
			c->dbits = (Byte)bd;
			c->ltree = tl;
			c->dtree = td;
			LuTracev((stderr, "inflate:       codes new\n"));
		}
		return c;
	}


	inline int inflate_codes(inflate_blocks_statef *s, z_streamp z, int r) {
		uInt j;               // temporary storage
		const inflate_huft *t;      // temporary pointer
		uInt e;               // extra bits or operation
		uLong b;              // bit buffer
		uInt k;               // bits in bit buffer
		Byte *p;             // input data pointer
		uInt n;               // bytes available there
		Byte *q;             // output window write pointer
		uInt m;               // bytes to end of window or read pointer
		Byte *f;             // pointer to copy strings from
		inflate_codes_statef *c = s->sub.decode.codes;  // codes state

														// copy input/output information to locals (UPDATE macro restores)
		LOAD

			// process input and output based on current state
			for (;;) switch (c->mode) {             // waiting for "i:"=input, "o:"=output, "x:"=nothing
				case START:         // x: set up for LEN
#ifndef SLOW
					if (m >= 258 && n >= 10) {
						UPDATE
							r = inflate_fast(c->lbits, c->dbits, c->ltree, c->dtree, s, z);
						LOAD
							if (r != Z_OK) {
								c->mode = r == Z_STREAM_END ? WASH : BADCODE;
								break;
							}
					}
#endif // !SLOW
					c->sub.code.need = c->lbits;
					c->sub.code.tree = c->ltree;
					c->mode = LEN;
				case LEN:           // i: get length/literal/eob next
					j = c->sub.code.need;
					NEEDBITS(j)
						t = c->sub.code.tree + ((uInt)b & inflate_mask[j]);
					DUMPBITS(t->Z_bits)
						e = (uInt)(t->exop);
					if (e == 0)               // literal 
					{
						c->sub.lit = t->base;
						LuTracevv((stderr, t->base >= 0x20 && t->base < 0x7f ?
								   "inflate:         literal '%c'\n" :
								   "inflate:         literal 0x%02x\n", t->base));
						c->mode = LIT;
						break;
					}
					if (e & 16)               // length 
					{
						c->sub.copy.get = e & 15;
						c->len = t->base;
						c->mode = LENEXT;
						break;
					}
					if ((e & 64) == 0)        // next table 
					{
						c->sub.code.need = e;
						c->sub.code.tree = t + t->base;
						break;
					}
					if (e & 32)               // end of block 
					{
						LuTracevv((stderr, "inflate:         end of block\n"));
						c->mode = WASH;
						break;
					}
					c->mode = BADCODE;        // invalid code 
					z->msg = (char*)"invalid literal/length code";
					r = Z_DATA_ERROR;
					LEAVE
				case LENEXT:        // i: getting length extra (have base) 
					j = c->sub.copy.get;
					NEEDBITS(j)
						c->len += (uInt)b & inflate_mask[j];
					DUMPBITS(j)
						c->sub.code.need = c->dbits;
					c->sub.code.tree = c->dtree;
					LuTracevv((stderr, "inflate:         length %u\n", c->len));
					c->mode = DIST;
				case DIST:          // i: get distance next 
					j = c->sub.code.need;
					NEEDBITS(j)
						t = c->sub.code.tree + ((uInt)b & inflate_mask[j]);
					DUMPBITS(t->Z_bits)
						e = (uInt)(t->exop);
					if (e & 16)               // distance 
					{
						c->sub.copy.get = e & 15;
						c->sub.copy.dist = t->base;
						c->mode = DISTEXT;
						break;
					}
					if ((e & 64) == 0)        // next table 
					{
						c->sub.code.need = e;
						c->sub.code.tree = t + t->base;
						break;
					}
					c->mode = BADCODE;        // invalid code 
					z->msg = (char*)"invalid distance code";
					r = Z_DATA_ERROR;
					LEAVE
				case DISTEXT:       // i: getting distance extra 
					j = c->sub.copy.get;
					NEEDBITS(j)
						c->sub.copy.dist += (uInt)b & inflate_mask[j];
					DUMPBITS(j)
						LuTracevv((stderr, "inflate:         distance %u\n", c->sub.copy.dist));
					c->mode = COPY;
				case COPY:          // o: copying bytes in window, waiting for space 
					f = q - c->sub.copy.dist;
					while (f < s->window)             // modulo window size-"while" instead
						f += s->end - s->window;        // of "if" handles invalid distances 
					while (c->len) {
						NEEDOUT
							OUTBYTE(*f++)
							if (f == s->end)
								f = s->window;
						c->len--;
					}
					c->mode = START;
					break;
				case LIT:           // o: got literal, waiting for output space 
					NEEDOUT
						OUTBYTE(c->sub.lit)
						c->mode = START;
					break;
				case WASH:          // o: got eob, possibly more output 
					if (k > 7)        // return unused byte, if any 
					{
						//Assert(k < 16, "inflate_codes grabbed too many bytes")
						k -= 8;
						n++;
						p--;            // can always return one 
					}
					FLUSH
						if (s->read != s->write)
							LEAVE
							c->mode = END;
				case END:
					r = Z_STREAM_END;
					LEAVE
				case BADCODE:       // x: got error
					r = Z_DATA_ERROR;
					LEAVE
				default:
					r = Z_STREAM_ERROR;
					LEAVE
			}
	}


	inline void inflate_codes_free(inflate_codes_statef *c, z_streamp z) {
		ZFREE(z, c);
		LuTracev((stderr, "inflate:       codes free\n"));
	}



	// infblock.c -- interpret and process block types to last block
	// Copyright (C) 1995-1998 Mark Adler
	// For conditions of distribution and use, see copyright notice in zlib.h

	//struct inflate_codes_state {int dummy;}; // for buggy compilers 



	// Table for deflate from PKZIP's appnote.txt.
	const uInt border[] = { // Order of the bit length code lengths
		16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

	//
	// Notes beyond the 1.93a appnote.txt:
	//
	// 1. Distance pointers never point before the beginning of the output stream.
	// 2. Distance pointers can point back across blocks, up to 32k away.
	// 3. There is an implied maximum of 7 bits for the bit length table and
	//    15 bits for the actual data.
	// 4. If only one code exists, then it is encoded using one bit.  (Zero
	//    would be more efficient, but perhaps a little confusing.)  If two
	//    codes exist, they are coded using one bit each (0 and 1).
	// 5. There is no way of sending zero distance codes--a dummy must be
	//    sent if there are none.  (History: a pre 2.0 version of PKZIP would
	//    store blocks with no distance codes, but this was discovered to be
	//    too harsh a criterion.)  Valid only for 1.93a.  2.04c does allow
	//    zero distance codes, which is sent as one code of zero bits in
	//    length.
	// 6. There are up to 286 literal/length codes.  Code 256 represents the
	//    end-of-block.  Note however that the static length tree defines
	//    288 codes just to fill out the Huffman codes.  Codes 286 and 287
	//    cannot be used though, since there is no length base or extra bits
	//    defined for them.  Similarily, there are up to 30 distance codes.
	//    However, static trees define 32 codes (all 5 bits) to fill out the
	//    Huffman codes, but the last two had better not show up in the data.
	// 7. Unzip can check dynamic Huffman blocks for complete code sets.
	//    The exception is that a single code would not be complete (see #4).
	// 8. The five bits following the block type is really the number of
	//    literal codes sent minus 257.
	// 9. Length codes 8,16,16 are interpreted as 13 length codes of 8 bits
	//    (1+6+6).  Therefore, to output three times the length, you output
	//    three codes (1+1+1), whereas to output four times the same length,
	//    you only need two codes (1+3).  Hmm.
	//10. In the tree reconstruction algorithm, Code = Code + Increment
	//    only if BitLength(i) is not zero.  (Pretty obvious.)
	//11. Correction: 4 Bits: # of Bit Length codes - 4     (4 - 19)
	//12. Note: length code 284 can represent 227-258, but length code 285
	//    really is 258.  The last length deserves its own, short code
	//    since it gets used a lot in very redundant files.  The length
	//    258 is special since 258 - 3 (the min match length) is 255.
	//13. The literal/length and distance code bit lengths are read as a
	//    single stream of lengths.  It is possible (and advantageous) for
	//    a repeat code (16, 17, or 18) to go across the boundary between
	//    the two sets of lengths.


	inline void inflate_blocks_reset(inflate_blocks_statef *s, z_streamp z, uLong *c) {
		if (c != Z_NULL)
			*c = s->check;
		if (s->mode == IBM_BTREE || s->mode == IBM_DTREE)
			ZFREE(z, s->sub.trees.blens);
		if (s->mode == IBM_CODES)
			inflate_codes_free(s->sub.decode.codes, z);
		s->mode = IBM_TYPE;
		s->bitk = 0;
		s->bitb = 0;
		s->read = s->write = s->window;
		if (s->checkfn != Z_NULL)
			z->adler = s->check = (*s->checkfn)(0L, (const Byte *)Z_NULL, 0);
		LuTracev((stderr, "inflate:   blocks reset\n"));
	}


	inline inflate_blocks_statef *inflate_blocks_new(z_streamp z, check_func c, uInt w) {
		inflate_blocks_statef *s;

		if ((s = (inflate_blocks_statef *)ZALLOC
		(z, 1, sizeof(struct inflate_blocks_state))) == Z_NULL)
			return s;
		if ((s->hufts =
			(inflate_huft *)ZALLOC(z, sizeof(inflate_huft), MANY)) == Z_NULL) {
			ZFREE(z, s);
			return Z_NULL;
		}
		if ((s->window = (Byte *)ZALLOC(z, 1, w)) == Z_NULL) {
			ZFREE(z, s->hufts);
			ZFREE(z, s);
			return Z_NULL;
		}
		s->end = s->window + w;
		s->checkfn = c;
		s->mode = IBM_TYPE;
		LuTracev((stderr, "inflate:   blocks allocated\n"));
		inflate_blocks_reset(s, z, Z_NULL);
		return s;
	}


	inline int inflate_blocks(inflate_blocks_statef *s, z_streamp z, int r) {
		uInt t;               // temporary storage
		uLong b;              // bit buffer
		uInt k;               // bits in bit buffer
		Byte *p;             // input data pointer
		uInt n;               // bytes available there
		Byte *q;             // output window write pointer
		uInt m;               // bytes to end of window or read pointer 

							  // copy input/output information to locals (UPDATE macro restores) 
		LOAD

			// process input based on current state 
			for (;;) switch (s->mode) {
				case IBM_TYPE:
					NEEDBITS(3)
						t = (uInt)b & 7;
					s->last = t & 1;
					switch (t >> 1) {
						case 0:                         // stored 
							LuTracev((stderr, "inflate:     stored block%s\n",
									  s->last ? " (last)" : ""));
							DUMPBITS(3)
								t = k & 7;                    // go to byte boundary 
							DUMPBITS(t)
								s->mode = IBM_LENS;               // get length of stored block
							break;
						case 1:                         // fixed 
							LuTracev((stderr, "inflate:     fixed codes block%s\n",
									  s->last ? " (last)" : ""));
							{
								uInt bl, bd;
								const inflate_huft *tl, *td;

								inflate_trees_fixed(&bl, &bd, &tl, &td, z);
								s->sub.decode.codes = inflate_codes_new(bl, bd, tl, td, z);
								if (s->sub.decode.codes == Z_NULL) {
									r = Z_MEM_ERROR;
									LEAVE
								}
							}
							DUMPBITS(3)
								s->mode = IBM_CODES;
							break;
						case 2:                         // dynamic 
							LuTracev((stderr, "inflate:     dynamic codes block%s\n",
									  s->last ? " (last)" : ""));
							DUMPBITS(3)
								s->mode = IBM_TABLE;
							break;
						case 3:                         // illegal
							DUMPBITS(3)
								s->mode = IBM_BAD;
							z->msg = (char*)"invalid block type";
							r = Z_DATA_ERROR;
							LEAVE
					}
					break;
				case IBM_LENS:
					NEEDBITS(32)
						if ((((~b) >> 16) & 0xffff) != (b & 0xffff)) {
							s->mode = IBM_BAD;
							z->msg = (char*)"invalid stored block lengths";
							r = Z_DATA_ERROR;
							LEAVE
						}
					s->sub.left = (uInt)b & 0xffff;
					b = k = 0;                      // dump bits 
					LuTracev((stderr, "inflate:       stored length %u\n", s->sub.left));
					s->mode = s->sub.left ? IBM_STORED : (s->last ? IBM_DRY : IBM_TYPE);
					break;
				case IBM_STORED:
					if (n == 0)
						LEAVE
						NEEDOUT
						t = s->sub.left;
					if (t > n) t = n;
					if (t > m) t = m;
					memcpy(q, p, t);
					p += t;  n -= t;
					q += t;  m -= t;
					if ((s->sub.left -= t) != 0)
						break;
					LuTracev((stderr, "inflate:       stored end, %lu total out\n",
							  z->total_out + (q >= s->read ? q - s->read :
							  (s->end - s->read) + (q - s->window))));
					s->mode = s->last ? IBM_DRY : IBM_TYPE;
					break;
				case IBM_TABLE:
					NEEDBITS(14)
						s->sub.trees.table = t = (uInt)b & 0x3fff;
					// remove this section to workaround bug in pkzip
					if ((t & 0x1f) > 29 || ((t >> 5) & 0x1f) > 29) {
						s->mode = IBM_BAD;
						z->msg = (char*)"too many length or distance symbols";
						r = Z_DATA_ERROR;
						LEAVE
					}
					// end remove
					t = 258 + (t & 0x1f) + ((t >> 5) & 0x1f);
					if ((s->sub.trees.blens = (uInt*)ZALLOC(z, t, sizeof(uInt))) == Z_NULL) {
						r = Z_MEM_ERROR;
						LEAVE
					}
					DUMPBITS(14)
						s->sub.trees.index = 0;
					LuTracev((stderr, "inflate:       table sizes ok\n"));
					s->mode = IBM_BTREE;
				case IBM_BTREE:
					while (s->sub.trees.index < 4 + (s->sub.trees.table >> 10)) {
						NEEDBITS(3)
							s->sub.trees.blens[border[s->sub.trees.index++]] = (uInt)b & 7;
						DUMPBITS(3)
					}
					while (s->sub.trees.index < 19)
						s->sub.trees.blens[border[s->sub.trees.index++]] = 0;
					s->sub.trees.bb = 7;
					t = inflate_trees_bits(s->sub.trees.blens, &s->sub.trees.bb,
										   &s->sub.trees.tb, s->hufts, z);
					if (t != Z_OK) {
						r = t;
						if (r == Z_DATA_ERROR) {
							ZFREE(z, s->sub.trees.blens);
							s->mode = IBM_BAD;
						}
						LEAVE
					}
					s->sub.trees.index = 0;
					LuTracev((stderr, "inflate:       bits tree ok\n"));
					s->mode = IBM_DTREE;
				case IBM_DTREE:
					while (t = s->sub.trees.table,
						   s->sub.trees.index < 258 + (t & 0x1f) + ((t >> 5) & 0x1f)) {
						inflate_huft *h;
						uInt i, j, c;

						t = s->sub.trees.bb;
						NEEDBITS(t)
							h = s->sub.trees.tb + ((uInt)b & inflate_mask[t]);
						t = h->Z_bits;
						c = h->base;
						if (c < 16) {
							DUMPBITS(t)
								s->sub.trees.blens[s->sub.trees.index++] = c;
						} else // c == 16..18 
						{
							i = c == 18 ? 7 : c - 14;
							j = c == 18 ? 11 : 3;
							NEEDBITS(t + i)
								DUMPBITS(t)
								j += (uInt)b & inflate_mask[i];
							DUMPBITS(i)
								i = s->sub.trees.index;
							t = s->sub.trees.table;
							if (i + j > 258 + (t & 0x1f) + ((t >> 5) & 0x1f) ||
								(c == 16 && i < 1)) {
								ZFREE(z, s->sub.trees.blens);
								s->mode = IBM_BAD;
								z->msg = (char*)"invalid bit length repeat";
								r = Z_DATA_ERROR;
								LEAVE
							}
							c = c == 16 ? s->sub.trees.blens[i - 1] : 0;
							do {
								s->sub.trees.blens[i++] = c;
							} while (--j);
							s->sub.trees.index = i;
						}
					}
					s->sub.trees.tb = Z_NULL;
					{
						uInt bl, bd;
						inflate_huft *tl, *td;
						inflate_codes_statef *c;

						bl = 9;         // must be <= 9 for lookahead assumptions 
						bd = 6;         // must be <= 9 for lookahead assumptions
						t = s->sub.trees.table;
						t = inflate_trees_dynamic(257 + (t & 0x1f), 1 + ((t >> 5) & 0x1f),
												  s->sub.trees.blens, &bl, &bd, &tl, &td,
												  s->hufts, z);
						if (t != Z_OK) {
							if (t == (uInt)Z_DATA_ERROR) {
								ZFREE(z, s->sub.trees.blens);
								s->mode = IBM_BAD;
							}
							r = t;
							LEAVE
						}
						LuTracev((stderr, "inflate:       trees ok\n"));
						if ((c = inflate_codes_new(bl, bd, tl, td, z)) == Z_NULL) {
							r = Z_MEM_ERROR;
							LEAVE
						}
						s->sub.decode.codes = c;
					}
					ZFREE(z, s->sub.trees.blens);
					s->mode = IBM_CODES;
				case IBM_CODES:
					UPDATE
						if ((r = inflate_codes(s, z, r)) != Z_STREAM_END)
							return inflate_flush(s, z, r);
					r = Z_OK;
					inflate_codes_free(s->sub.decode.codes, z);
					LOAD
						LuTracev((stderr, "inflate:       codes end, %lu total out\n",
								  z->total_out + (q >= s->read ? q - s->read :
								  (s->end - s->read) + (q - s->window))));
					if (!s->last) {
						s->mode = IBM_TYPE;
						break;
					}
					s->mode = IBM_DRY;
				case IBM_DRY:
					FLUSH
						if (s->read != s->write)
							LEAVE
							s->mode = IBM_DONE;
				case IBM_DONE:
					r = Z_STREAM_END;
					LEAVE
				case IBM_BAD:
					r = Z_DATA_ERROR;
					LEAVE
				default:
					r = Z_STREAM_ERROR;
					LEAVE
			}
	}


	inline int inflate_blocks_free(inflate_blocks_statef *s, z_streamp z) {
		inflate_blocks_reset(s, z, Z_NULL);
		ZFREE(z, s->window);
		ZFREE(z, s->hufts);
		ZFREE(z, s);
		LuTracev((stderr, "inflate:   blocks freed\n"));
		return Z_OK;
	}



	// inftrees.c -- generate Huffman trees for efficient decoding
	// Copyright (C) 1995-1998 Mark Adler
	// For conditions of distribution and use, see copyright notice in zlib.h
	//



	const char inflate_copyright[] =
		" inflate 1.1.3 Copyright 1995-1998 Mark Adler ";
	// If you use the zlib library in a product, an acknowledgment is welcome
	// in the documentation of your product. If for some reason you cannot
	// include such an acknowledgment, I would appreciate that you keep this
	// copyright string in the executable of your product.



	inline int huft_build(
		uInt *,            // code lengths in bits
		uInt,               // number of codes
		uInt,               // number of "simple" codes
		const uInt *,      // list of base values for non-simple codes
		const uInt *,      // list of extra bits for non-simple codes
		inflate_huft **,// result: starting table
		uInt *,            // maximum lookup bits (returns actual) 
		inflate_huft *,     // space for trees 
		uInt *,             // hufts used in space 
		uInt *);         // space for values 

						 // Tables for deflate from PKZIP's appnote.txt. 
	const uInt cplens[31] = { // Copy lengths for literal codes 257..285
		3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
		35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0 };
	// see note #13 above about 258
	const uInt cplext[31] = { // Extra bits for literal codes 257..285
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
		3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 112, 112 }; // 112==invalid
	const uInt cpdist[30] = { // Copy offsets for distance codes 0..29
		1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
		257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
		8193, 12289, 16385, 24577 };
	const uInt cpdext[30] = { // Extra bits for distance codes 
		0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
		7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
		12, 12, 13, 13 };

	//
	//   Huffman code decoding is performed using a multi-level table lookup.
	//   The fastest way to decode is to simply build a lookup table whose
	//   size is determined by the longest code.  However, the time it takes
	//   to build this table can also be a factor if the data being decoded
	//   is not very long.  The most common codes are necessarily the
	//   shortest codes, so those codes dominate the decoding time, and hence
	//   the speed.  The idea is you can have a shorter table that decodes the
	//   shorter, more probable codes, and then point to subsidiary tables for
	//   the longer codes.  The time it costs to decode the longer codes is
	//   then traded against the time it takes to make longer tables.
	//
	//   This results of this trade are in the variables lbits and dbits
	//   below.  lbits is the number of bits the first level table for literal/
	//   length codes can decode in one step, and dbits is the same thing for
	//   the distance codes.  Subsequent tables are also less than or equal to
	//   those sizes.  These values may be adjusted either when all of the
	//   codes are shorter than that, in which case the longest code length in
	//   bits is used, or when the shortest code is *longer* than the requested
	//   table size, in which case the length of the shortest code in bits is
	//   used.
	//
	//   There are two different values for the two tables, since they code a
	//   different number of possibilities each.  The literal/length table
	//   codes 286 possible values, or in a flat code, a little over eight
	//   bits.  The distance table codes 30 possible values, or a little less
	//   than five bits, flat.  The optimum values for speed end up being
	//   about one bit more than those, so lbits is 8+1 and dbits is 5+1.
	//   The optimum values may differ though from machine to machine, and
	//   possibly even between compilers.  Your mileage may vary.
	//


	// If BMAX needs to be larger than 16, then h and x[] should be uLong. 
#define BMAX 15         // maximum bit length of any code

	inline int huft_build(
		uInt *b,               // code lengths in bits (all assumed <= BMAX)
		uInt n,                 // number of codes (assumed <= 288)
		uInt s,                 // number of simple-valued codes (0..s-1)
		const uInt *d,         // list of base values for non-simple codes
		const uInt *e,         // list of extra bits for non-simple codes
		inflate_huft * *t,  // result: starting table
		uInt *m,               // maximum lookup bits, returns actual
		inflate_huft *hp,       // space for trees
		uInt *hn,               // hufts used in space
		uInt *v)               // working area: values in order of bit length
							   // Given a list of code lengths and a maximum table size, make a set of
							   // tables to decode that set of codes.  Return Z_OK on success, Z_BUF_ERROR
							   // if the given code set is incomplete (the tables are still built in this
							   // case), or Z_DATA_ERROR if the input is invalid.
	{

		uInt a;                       // counter for codes of length k
		uInt c[BMAX + 1];               // bit length count table
		uInt f;                       // i repeats in table every f entries 
		int g;                        // maximum code length 
		int h;                        // table level 
		register uInt i;              // counter, current code 
		register uInt j;              // counter
		register int k;               // number of bits in current code 
		int l;                        // bits per table (returned in m) 
		uInt mask;                    // (1 << w) - 1, to avoid cc -O bug on HP 
		register uInt *p;            // pointer into c[], b[], or v[]
		inflate_huft *q;              // points to current table 
		struct inflate_huft_s r;      // table entry for structure assignment 
		inflate_huft *u[BMAX];        // table stack 
		register int w;               // bits before this table == (l * h) 
		uInt x[BMAX + 1];               // bit offsets, then code stack 
		uInt *xp;                    // pointer into x 
		int y;                        // number of dummy codes added 
		uInt z;                       // number of entries in current table 


									  // Generate counts for each bit length 
		p = c;
#define C0 *p++ = 0;
#define C2 C0 C0 C0 C0
#define C4 C2 C2 C2 C2
		C4; p;                          // clear c[]--assume BMAX+1 is 16
		p = b;  i = n;
		do {
			c[*p++]++;                  // assume all entries <= BMAX 
		} while (--i);
		if (c[0] == n)                // null input--all zero length codes 
		{
			*t = (inflate_huft *)Z_NULL;
			*m = 0;
			return Z_OK;
		}


		// Find minimum and maximum length, bound *m by those 
		l = *m;
		for (j = 1; j <= BMAX; j++)
			if (c[j])
				break;
		k = j;                        // minimum code length 
		if ((uInt)l < j)
			l = j;
		for (i = BMAX; i; i--)
			if (c[i])
				break;
		g = i;                        // maximum code length 
		if ((uInt)l > i)
			l = i;
		*m = l;


		// Adjust last length count to fill out codes, if needed 
		for (y = 1 << j; j < i; j++, y <<= 1)
			if ((y -= c[j]) < 0)
				return Z_DATA_ERROR;
		if ((y -= c[i]) < 0)
			return Z_DATA_ERROR;
		c[i] += y;


		// Generate starting offsets into the value table for each length 
		x[1] = j = 0;
		p = c + 1;  xp = x + 2;
		while (--i) {                 // note that i == g from above 
			*xp++ = (j += *p++);
		}


		// Make a table of values in order of bit lengths 
		p = b;  i = 0;
		do {
			if ((j = *p++) != 0)
				v[x[j]++] = i;
		} while (++i < n);
		n = x[g];                     // set n to length of v 


									  // Generate the Huffman codes and for each, make the table entries 
		x[0] = i = 0;                 // first Huffman code is zero 
		p = v;                        // grab values in bit order 
		h = -1;                       // no tables yet--level -1 
		w = -l;                       // bits decoded == (l * h) 
		u[0] = (inflate_huft *)Z_NULL;        // just to keep compilers happy 
		q = (inflate_huft *)Z_NULL;   // ditto 
		z = 0;                        // ditto 

									  // go through the bit lengths (k already is bits in shortest code) 
		for (; k <= g; k++) {
			a = c[k];
			while (a--) {
				// here i is the Huffman code of length k bits for value *p 
				// make tables up to required level 
				while (k > w + l) {
					h++;
					w += l;                 // previous table always l bits 

											// compute minimum size table less than or equal to l bits
					z = g - w;
					z = z > (uInt)l ? l : z;        // table size upper limit 
					if ((f = 1 << (j = k - w)) > a + 1)     // try a k-w bit table 
					{                       // too few codes for k-w bit table 
						f -= a + 1;           // deduct codes from patterns left 
						xp = c + k;
						if (j < z)
							while (++j < z)     // try smaller tables up to z bits 
							{
								if ((f <<= 1) <= *++xp)
									break;          // enough codes to use up j bits 
								f -= *xp;         // else deduct codes from patterns
							}
					}
					z = 1 << j;             // table entries for j-bit table 

											// allocate new table 
					if (*hn + z > MANY)     // (note: doesn't matter for fixed) 
						return Z_DATA_ERROR;  // overflow of MANY 
					u[h] = q = hp + *hn;
					*hn += z;

					// connect to last table, if there is one 
					if (h) {
						x[h] = i;             // save pattern for backing up
						r.Z_bits = (Byte)l;     // bits to dump before this table 
						r.exop = (Byte)j;     // bits in this table 
						j = i >> (w - l);
						r.base = (uInt)(q - u[h - 1] - j);   // offset to this table 
						u[h - 1][j] = r;        // connect to last table 
					} else
						*t = q;               // first table is returned result 
				}

				// set up table entry in r 
				r.Z_bits = (Byte)(k - w);
				if (p >= v + n)
					r.exop = 128 + 64;      // out of values--invalid code 
				else if (*p < s) {
					r.exop = (Byte)(*p < 256 ? 0 : 32 + 64);     // 256 is end-of-block 
					r.base = *p++;          // simple code is just the value 
				} else {
					r.exop = (Byte)(e[*p - s] + 16 + 64);// non-simple--look up in lists 
					r.base = d[*p++ - s];
				}

				// fill code-like entries with r
				f = 1 << (k - w);
				for (j = i >> w; j < z; j += f)
					q[j] = r;

				// backwards increment the k-bit code i 
				for (j = 1 << (k - 1); i & j; j >>= 1)
					i ^= j;
				i ^= j;

				// backup over finished tables 
				mask = (1 << w) - 1;      // needed on HP, cc -O bug 
				while ((i & mask) != x[h]) {
					h--;                    // don't need to update q
					w -= l;
					mask = (1 << w) - 1;
				}
			}
		}


		// Return Z_BUF_ERROR if we were given an incomplete table 
		return y != 0 && g != 1 ? Z_BUF_ERROR : Z_OK;
	}


	inline int inflate_trees_bits(
		uInt *c,               // 19 code lengths
		uInt *bb,              // bits tree desired/actual depth
		inflate_huft * *tb, // bits tree result
		inflate_huft *hp,       // space for trees
		z_streamp z)            // for messages
	{
		int r;
		uInt hn = 0;          // hufts used in space 
		uInt *v;             // work area for huft_build 

		if ((v = (uInt*)ZALLOC(z, 19, sizeof(uInt))) == Z_NULL)
			return Z_MEM_ERROR;
		r = huft_build(c, 19, 19, (uInt*)Z_NULL, (uInt*)Z_NULL,
					   tb, bb, hp, &hn, v);
		if (r == Z_DATA_ERROR)
			z->msg = (char*)"oversubscribed dynamic bit lengths tree";
		else if (r == Z_BUF_ERROR || *bb == 0) {
			z->msg = (char*)"incomplete dynamic bit lengths tree";
			r = Z_DATA_ERROR;
		}
		ZFREE(z, v);
		return r;
	}


	inline int inflate_trees_dynamic(
		uInt nl,                // number of literal/length codes
		uInt nd,                // number of distance codes
		uInt *c,               // that many (total) code lengths
		uInt *bl,              // literal desired/actual bit depth
		uInt *bd,              // distance desired/actual bit depth
		inflate_huft * *tl, // literal/length tree result
		inflate_huft * *td, // distance tree result
		inflate_huft *hp,       // space for trees
		z_streamp z)            // for messages
	{
		int r;
		uInt hn = 0;          // hufts used in space 
		uInt *v;             // work area for huft_build 

							 // allocate work area 
		if ((v = (uInt*)ZALLOC(z, 288, sizeof(uInt))) == Z_NULL)
			return Z_MEM_ERROR;

		// build literal/length tree 
		r = huft_build(c, nl, 257, cplens, cplext, tl, bl, hp, &hn, v);
		if (r != Z_OK || *bl == 0) {
			if (r == Z_DATA_ERROR)
				z->msg = (char*)"oversubscribed literal/length tree";
			else if (r != Z_MEM_ERROR) {
				z->msg = (char*)"incomplete literal/length tree";
				r = Z_DATA_ERROR;
			}
			ZFREE(z, v);
			return r;
		}

		// build distance tree 
		r = huft_build(c + nl, nd, 0, cpdist, cpdext, td, bd, hp, &hn, v);
		if (r != Z_OK || (*bd == 0 && nl > 257)) {
			if (r == Z_DATA_ERROR)
				z->msg = (char*)"oversubscribed distance tree";
			else if (r == Z_BUF_ERROR) {
				z->msg = (char*)"incomplete distance tree";
				r = Z_DATA_ERROR;
			} else if (r != Z_MEM_ERROR) {
				z->msg = (char*)"empty distance tree with lengths";
				r = Z_DATA_ERROR;
			}
			ZFREE(z, v);
			return r;
		}

		// done 
		ZFREE(z, v);
		return Z_OK;
	}





	inline int inflate_trees_fixed(
		uInt *bl,               // literal desired/actual bit depth
		uInt *bd,               // distance desired/actual bit depth
		const inflate_huft * * tl,     // literal/length tree result
		const inflate_huft * *td,     // distance tree result
		z_streamp)             // for memory allocation
	{
		*bl = fixed_bl;
		*bd = fixed_bd;
		*tl = fixed_tl;
		*td = fixed_td;
		return Z_OK;
	}


	// inffast.c -- process literals and length/distance pairs fast
	// Copyright (C) 1995-1998 Mark Adler
	// For conditions of distribution and use, see copyright notice in zlib.h
	//


	//struct inflate_codes_state {int dummy;}; // for buggy compilers 


	// macros for bit input with no checking and for returning unused bytes 
#define GRABBITS(j) {while(k<(j)){b|=((uLong)NEXTBYTE)<<k;k+=8;}}
#define UNGRAB {c=z->avail_in-n;c=(k>>3)<c?k>>3:c;n+=c;p-=c;k-=c<<3;}

	// Called with number of bytes left to write in window at least 258
	// (the maximum string length) and number of input bytes available
	// at least ten.  The ten bytes are six bytes for the longest length/
	// distance pair plus four bytes for overloading the bit buffer. 

	inline int inflate_fast(
		uInt bl, uInt bd,
		const inflate_huft *tl,
		const inflate_huft *td, // need separate declaration for Borland C++
		inflate_blocks_statef *s,
		z_streamp z) {
		const inflate_huft *t;      // temporary pointer 
		uInt e;               // extra bits or operation 
		uLong b;              // bit buffer 
		uInt k;               // bits in bit buffer 
		Byte *p;             // input data pointer 
		uInt n;               // bytes available there 
		Byte *q;             // output window write pointer 
		uInt m;               // bytes to end of window or read pointer 
		uInt ml;              // mask for literal/length tree
		uInt md;              // mask for distance tree 
		uInt c;               // bytes to copy 
		uInt d;               // distance back to copy from 
		Byte *r;             // copy source pointer 

							 // load input, output, bit values 
		LOAD

			// initialize masks 
			ml = inflate_mask[bl];
		md = inflate_mask[bd];

		// do until not enough input or output space for fast loop 
		do {                          // assume called with m >= 258 && n >= 10 
									  // get literal/length code 
			GRABBITS(20)                // max bits for literal/length code 
				if ((e = (t = tl + ((uInt)b & ml))->exop) == 0) {
					DUMPBITS(t->Z_bits)
						LuTracevv((stderr, t->base >= 0x20 && t->base < 0x7f ?
								   "inflate:         * literal '%c'\n" :
								   "inflate:         * literal 0x%02x\n", t->base));
					*q++ = (Byte)t->base;
					m--;
					continue;
				}
			for (;;) {
				DUMPBITS(t->Z_bits)
					if (e & 16) {
						// get extra bits for length 
						e &= 15;
						c = t->base + ((uInt)b & inflate_mask[e]);
						DUMPBITS(e)
							LuTracevv((stderr, "inflate:         * length %u\n", c));

						// decode distance base of block to copy 
						GRABBITS(15);           // max bits for distance code 
						e = (t = td + ((uInt)b & md))->exop;
						for (;;) {
							DUMPBITS(t->Z_bits)
								if (e & 16) {
									// get extra bits to add to distance base 
									e &= 15;
									GRABBITS(e)         // get extra bits (up to 13) 
										d = t->base + ((uInt)b & inflate_mask[e]);
									DUMPBITS(e)
										LuTracevv((stderr, "inflate:         * distance %u\n", d));

									// do the copy
									m -= c;
									r = q - d;
									if (r < s->window)                  // wrap if needed
									{
										do {
											r += s->end - s->window;        // force pointer in window
										} while (r < s->window);          // covers invalid distances
										e = (uInt)(s->end - r);
										if (c > e) {
											c -= e;                         // wrapped copy
											do {
												*q++ = *r++;
											} while (--e);
											r = s->window;
											do {
												*q++ = *r++;
											} while (--c);
										} else                              // normal copy
										{
											*q++ = *r++;  c--;
											*q++ = *r++;  c--;
											do {
												*q++ = *r++;
											} while (--c);
										}
									} else                                /* normal copy */
									{
										*q++ = *r++;  c--;
										*q++ = *r++;  c--;
										do {
											*q++ = *r++;
										} while (--c);
									}
									break;
								} else if ((e & 64) == 0) {
									t += t->base;
									e = (t += ((uInt)b & inflate_mask[e]))->exop;
								} else {
									z->msg = (char*)"invalid distance code";
									UNGRAB
										UPDATE
										return Z_DATA_ERROR;
								}
						};
						break;
					}
				if ((e & 64) == 0) {
					t += t->base;
					if ((e = (t += ((uInt)b & inflate_mask[e]))->exop) == 0) {
						DUMPBITS(t->Z_bits)
							LuTracevv((stderr, t->base >= 0x20 && t->base < 0x7f ?
									   "inflate:         * literal '%c'\n" :
									   "inflate:         * literal 0x%02x\n", t->base));
						*q++ = (Byte)t->base;
						m--;
						break;
					}
				} else if (e & 32) {
					LuTracevv((stderr, "inflate:         * end of block\n"));
					UNGRAB
						UPDATE
						return Z_STREAM_END;
				} else {
					z->msg = (char*)"invalid literal/length code";
					UNGRAB
						UPDATE
						return Z_DATA_ERROR;
				}
			};
		} while (m >= 258 && n >= 10);

		// not enough input or output--restore pointers and return
		UNGRAB
			UPDATE
			return Z_OK;
	}






	// crc32.c -- compute the CRC-32 of a data stream
	// Copyright (C) 1995-1998 Mark Adler
	// For conditions of distribution and use, see copyright notice in zlib.h

	// @(#) $Id$






	// Table of CRC-32's of all single-byte values (made by make_crc_table)
	const uLong crc_table[256] = {
		0x00000000L, 0x77073096L, 0xee0e612cL, 0x990951baL, 0x076dc419L,
		0x706af48fL, 0xe963a535L, 0x9e6495a3L, 0x0edb8832L, 0x79dcb8a4L,
		0xe0d5e91eL, 0x97d2d988L, 0x09b64c2bL, 0x7eb17cbdL, 0xe7b82d07L,
		0x90bf1d91L, 0x1db71064L, 0x6ab020f2L, 0xf3b97148L, 0x84be41deL,
		0x1adad47dL, 0x6ddde4ebL, 0xf4d4b551L, 0x83d385c7L, 0x136c9856L,
		0x646ba8c0L, 0xfd62f97aL, 0x8a65c9ecL, 0x14015c4fL, 0x63066cd9L,
		0xfa0f3d63L, 0x8d080df5L, 0x3b6e20c8L, 0x4c69105eL, 0xd56041e4L,
		0xa2677172L, 0x3c03e4d1L, 0x4b04d447L, 0xd20d85fdL, 0xa50ab56bL,
		0x35b5a8faL, 0x42b2986cL, 0xdbbbc9d6L, 0xacbcf940L, 0x32d86ce3L,
		0x45df5c75L, 0xdcd60dcfL, 0xabd13d59L, 0x26d930acL, 0x51de003aL,
		0xc8d75180L, 0xbfd06116L, 0x21b4f4b5L, 0x56b3c423L, 0xcfba9599L,
		0xb8bda50fL, 0x2802b89eL, 0x5f058808L, 0xc60cd9b2L, 0xb10be924L,
		0x2f6f7c87L, 0x58684c11L, 0xc1611dabL, 0xb6662d3dL, 0x76dc4190L,
		0x01db7106L, 0x98d220bcL, 0xefd5102aL, 0x71b18589L, 0x06b6b51fL,
		0x9fbfe4a5L, 0xe8b8d433L, 0x7807c9a2L, 0x0f00f934L, 0x9609a88eL,
		0xe10e9818L, 0x7f6a0dbbL, 0x086d3d2dL, 0x91646c97L, 0xe6635c01L,
		0x6b6b51f4L, 0x1c6c6162L, 0x856530d8L, 0xf262004eL, 0x6c0695edL,
		0x1b01a57bL, 0x8208f4c1L, 0xf50fc457L, 0x65b0d9c6L, 0x12b7e950L,
		0x8bbeb8eaL, 0xfcb9887cL, 0x62dd1ddfL, 0x15da2d49L, 0x8cd37cf3L,
		0xfbd44c65L, 0x4db26158L, 0x3ab551ceL, 0xa3bc0074L, 0xd4bb30e2L,
		0x4adfa541L, 0x3dd895d7L, 0xa4d1c46dL, 0xd3d6f4fbL, 0x4369e96aL,
		0x346ed9fcL, 0xad678846L, 0xda60b8d0L, 0x44042d73L, 0x33031de5L,
		0xaa0a4c5fL, 0xdd0d7cc9L, 0x5005713cL, 0x270241aaL, 0xbe0b1010L,
		0xc90c2086L, 0x5768b525L, 0x206f85b3L, 0xb966d409L, 0xce61e49fL,
		0x5edef90eL, 0x29d9c998L, 0xb0d09822L, 0xc7d7a8b4L, 0x59b33d17L,
		0x2eb40d81L, 0xb7bd5c3bL, 0xc0ba6cadL, 0xedb88320L, 0x9abfb3b6L,
		0x03b6e20cL, 0x74b1d29aL, 0xead54739L, 0x9dd277afL, 0x04db2615L,
		0x73dc1683L, 0xe3630b12L, 0x94643b84L, 0x0d6d6a3eL, 0x7a6a5aa8L,
		0xe40ecf0bL, 0x9309ff9dL, 0x0a00ae27L, 0x7d079eb1L, 0xf00f9344L,
		0x8708a3d2L, 0x1e01f268L, 0x6906c2feL, 0xf762575dL, 0x806567cbL,
		0x196c3671L, 0x6e6b06e7L, 0xfed41b76L, 0x89d32be0L, 0x10da7a5aL,
		0x67dd4accL, 0xf9b9df6fL, 0x8ebeeff9L, 0x17b7be43L, 0x60b08ed5L,
		0xd6d6a3e8L, 0xa1d1937eL, 0x38d8c2c4L, 0x4fdff252L, 0xd1bb67f1L,
		0xa6bc5767L, 0x3fb506ddL, 0x48b2364bL, 0xd80d2bdaL, 0xaf0a1b4cL,
		0x36034af6L, 0x41047a60L, 0xdf60efc3L, 0xa867df55L, 0x316e8eefL,
		0x4669be79L, 0xcb61b38cL, 0xbc66831aL, 0x256fd2a0L, 0x5268e236L,
		0xcc0c7795L, 0xbb0b4703L, 0x220216b9L, 0x5505262fL, 0xc5ba3bbeL,
		0xb2bd0b28L, 0x2bb45a92L, 0x5cb36a04L, 0xc2d7ffa7L, 0xb5d0cf31L,
		0x2cd99e8bL, 0x5bdeae1dL, 0x9b64c2b0L, 0xec63f226L, 0x756aa39cL,
		0x026d930aL, 0x9c0906a9L, 0xeb0e363fL, 0x72076785L, 0x05005713L,
		0x95bf4a82L, 0xe2b87a14L, 0x7bb12baeL, 0x0cb61b38L, 0x92d28e9bL,
		0xe5d5be0dL, 0x7cdcefb7L, 0x0bdbdf21L, 0x86d3d2d4L, 0xf1d4e242L,
		0x68ddb3f8L, 0x1fda836eL, 0x81be16cdL, 0xf6b9265bL, 0x6fb077e1L,
		0x18b74777L, 0x88085ae6L, 0xff0f6a70L, 0x66063bcaL, 0x11010b5cL,
		0x8f659effL, 0xf862ae69L, 0x616bffd3L, 0x166ccf45L, 0xa00ae278L,
		0xd70dd2eeL, 0x4e048354L, 0x3903b3c2L, 0xa7672661L, 0xd06016f7L,
		0x4969474dL, 0x3e6e77dbL, 0xaed16a4aL, 0xd9d65adcL, 0x40df0b66L,
		0x37d83bf0L, 0xa9bcae53L, 0xdebb9ec5L, 0x47b2cf7fL, 0x30b5ffe9L,
		0xbdbdf21cL, 0xcabac28aL, 0x53b39330L, 0x24b4a3a6L, 0xbad03605L,
		0xcdd70693L, 0x54de5729L, 0x23d967bfL, 0xb3667a2eL, 0xc4614ab8L,
		0x5d681b02L, 0x2a6f2b94L, 0xb40bbe37L, 0xc30c8ea1L, 0x5a05df1bL,
		0x2d02ef8dL
	};

	inline const uLong * get_crc_table() {
		return (const uLong *)crc_table;
	}

#define CRC_DO1(buf) crc = crc_table[((int)crc ^ (*buf++)) & 0xff] ^ (crc >> 8);
#define CRC_DO2(buf)  CRC_DO1(buf); CRC_DO1(buf);
#define CRC_DO4(buf)  CRC_DO2(buf); CRC_DO2(buf);
#define CRC_DO8(buf)  CRC_DO4(buf); CRC_DO4(buf);

	inline uLong ucrc32(uLong crc, const Byte *buf, uInt len) {
		if (buf == Z_NULL) return 0L;
		crc = crc ^ 0xffffffffL;
		while (len >= 8) {
			CRC_DO8(buf); len -= 8;
		}
		if (len) do {
			CRC_DO1(buf);
		} while (--len);
		return crc ^ 0xffffffffL;
	}



	// =============================================================
	// some decryption routines
#define CRC32(c, b) (crc_table[((int)(c)^(b))&0xff]^((c)>>8))
	inline void Uupdate_keys(unsigned long *keys, char c) {
		keys[0] = CRC32(keys[0], c);
		keys[1] += keys[0] & 0xFF;
		keys[1] = keys[1] * 134775813L + 1;
		keys[2] = CRC32(keys[2], keys[1] >> 24);
	}
	inline char Udecrypt_byte(unsigned long *keys) {
		unsigned temp = ((unsigned)keys[2] & 0xffff) | 2;
		return (char)(((temp * (temp ^ 1)) >> 8) & 0xff);
	}
	inline char zdecode(unsigned long *keys, char c) {
		c ^= Udecrypt_byte(keys);
		Uupdate_keys(keys, c);
		return c;
	}



	// adler32.c -- compute the Adler-32 checksum of a data stream
	// Copyright (C) 1995-1998 Mark Adler
	// For conditions of distribution and use, see copyright notice in zlib.h

	// @(#) $Id$


#define BASE 65521L // largest prime smaller than 65536
#define NMAX 5552
	// NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1

#define AD_DO1(buf,i)  {s1 += buf[i]; s2 += s1;}
#define AD_DO2(buf,i)  AD_DO1(buf,i); AD_DO1(buf,i+1);
#define AD_DO4(buf,i)  AD_DO2(buf,i); AD_DO2(buf,i+2);
#define AD_DO8(buf,i)  AD_DO4(buf,i); AD_DO4(buf,i+4);
#define AD_DO16(buf)   AD_DO8(buf,0); AD_DO8(buf,8);

	// =========================================================================
	inline uLong adler32(uLong adler, const Byte *buf, uInt len) {
		unsigned long s1 = adler & 0xffff;
		unsigned long s2 = (adler >> 16) & 0xffff;
		int k;

		if (buf == Z_NULL) return 1L;

		while (len > 0) {
			k = len < NMAX ? len : NMAX;
			len -= k;
			while (k >= 16) {
				AD_DO16(buf);
				buf += 16;
				k -= 16;
			}
			if (k != 0) do {
				s1 += *buf++;
				s2 += s1;
			} while (--k);
			s1 %= BASE;
			s2 %= BASE;
		}
		return (s2 << 16) | s1;
	}



	// zutil.c -- target dependent utility functions for the compression library
	// Copyright (C) 1995-1998 Jean-loup Gailly.
	// For conditions of distribution and use, see copyright notice in zlib.h
	// @(#) $Id$






	inline const char * zlibVersion() {
		return ZLIB_VERSION;
	}

	// exported to allow conversion of error code to string for compress() and
	// uncompress()
	inline const char * zError(int err) {
		return ERR_MSG(err);
	}




	inline voidpf zcalloc(voidpf opaque, unsigned items, unsigned size) {
		if (opaque) items += size - size; // make compiler happy
		return (voidpf)calloc(items, size);
	}

	inline void  zcfree(voidpf opaque, voidpf ptr) {
		zfree(ptr);
		if (opaque) return; // make compiler happy
	}



	// inflate.c -- zlib interface to inflate modules
	// Copyright (C) 1995-1998 Mark Adler
	// For conditions of distribution and use, see copyright notice in zlib.h

	//struct inflate_blocks_state {int dummy;}; // for buggy compilers

	typedef enum {
		IM_METHOD,   // waiting for method byte
		IM_FLAG,     // waiting for flag byte
		IM_DICT4,    // four dictionary check bytes to go
		IM_DICT3,    // three dictionary check bytes to go
		IM_DICT2,    // two dictionary check bytes to go
		IM_DICT1,    // one dictionary check byte to go
		IM_DICT0,    // waiting for inflateSetDictionary
		IM_BLOCKS,   // decompressing blocks
		IM_CHECK4,   // four check bytes to go
		IM_CHECK3,   // three check bytes to go
		IM_CHECK2,   // two check bytes to go
		IM_CHECK1,   // one check byte to go
		IM_DONE,     // finished check, done
		IM_BAD
	}      // got an error--stay here
	inflate_mode;

	// inflate private state
	struct internal_state {

		// mode
		inflate_mode  mode;   // current inflate mode

							  // mode dependent information
		union {
			uInt method;        // if IM_FLAGS, method byte
			struct {
				uLong was;                // computed check value
				uLong need;               // stream check value
			} check;            // if CHECK, check values to compare
			uInt marker;        // if IM_BAD, inflateSync's marker bytes count
		} sub;        // submode

					  // mode independent information
		int  nowrap;          // flag for no wrapper
		uInt wbits;           // log2(window size)  (8..15, defaults to 15)
		inflate_blocks_statef
			*blocks;            // current inflate_blocks state

	};

	inline int inflateReset(z_streamp z) {
		if (z == Z_NULL || z->state == Z_NULL)
			return Z_STREAM_ERROR;
		z->total_in = z->total_out = 0;
		z->msg = Z_NULL;
		z->state->mode = z->state->nowrap ? IM_BLOCKS : IM_METHOD;
		inflate_blocks_reset(z->state->blocks, z, Z_NULL);
		LuTracev((stderr, "inflate: reset\n"));
		return Z_OK;
	}

	inline int inflateEnd(z_streamp z) {
		if (z == Z_NULL || z->state == Z_NULL || z->zfree == Z_NULL)
			return Z_STREAM_ERROR;
		if (z->state->blocks != Z_NULL)
			inflate_blocks_free(z->state->blocks, z);
		ZFREE(z, z->state);
		z->state = Z_NULL;
		LuTracev((stderr, "inflate: end\n"));
		return Z_OK;
	}


	inline int inflateInit2(z_streamp z) {
		const char *version = ZLIB_VERSION; int stream_size = sizeof(z_stream);
		if (version == Z_NULL || version[0] != ZLIB_VERSION[0] || stream_size != sizeof(z_stream)) return Z_VERSION_ERROR;

		int w = -15; // MAX_WBITS: 32K LZ77 window.
					 // Warning: reducing MAX_WBITS makes minigzip unable to extract .gz files created by gzip.
					 // The memory requirements for deflate are (in bytes):
					 //            (1 << (windowBits+2)) +  (1 << (memLevel+9))
					 // that is: 128K for windowBits=15  +  128K for memLevel = 8  (default values)
					 // plus a few kilobytes for small objects. For example, if you want to reduce
					 // the default memory requirements from 256K to 128K, compile with
					 //     make CFLAGS="-O -DMAX_WBITS=14 -DMAX_MEM_LEVEL=7"
					 // Of course this will generally degrade compression (there's no free lunch).
					 //
					 //   The memory requirements for inflate are (in bytes) 1 << windowBits
					 // that is, 32K for windowBits=15 (default value) plus a few kilobytes
					 // for small objects.

					 // initialize state
		if (z == Z_NULL) return Z_STREAM_ERROR;
		z->msg = Z_NULL;
		if (z->zalloc == Z_NULL) {
			z->zalloc = zcalloc;
			z->opaque = (voidpf)0;
		}
		if (z->zfree == Z_NULL) z->zfree = zcfree;
		if ((z->state = (struct internal_state *)
			 ZALLOC(z, 1, sizeof(struct internal_state))) == Z_NULL)
			return Z_MEM_ERROR;
		z->state->blocks = Z_NULL;

		// handle undocumented nowrap option (no zlib header or check)
		z->state->nowrap = 0;
		if (w < 0) {
			w = -w;
			z->state->nowrap = 1;
		}

		// set window size
		if (w < 8 || w > 15) {
			inflateEnd(z);
			return Z_STREAM_ERROR;
		}
		z->state->wbits = (uInt)w;

		// create inflate_blocks state
		if ((z->state->blocks =
			 inflate_blocks_new(z, z->state->nowrap ? Z_NULL : adler32, (uInt)1 << w))
			== Z_NULL) {
			inflateEnd(z);
			return Z_MEM_ERROR;
		}
		LuTracev((stderr, "inflate: allocated\n"));

		// reset state 
		inflateReset(z);
		return Z_OK;
	}



#define IM_NEEDBYTE {if(z->avail_in==0)return r;r=f;}
#define IM_NEXTBYTE (z->avail_in--,z->total_in++,*z->next_in++)

	inline int inflate(z_streamp z, int f) {
		int r;
		uInt b;

		if (z == Z_NULL || z->state == Z_NULL || z->next_in == Z_NULL)
			return Z_STREAM_ERROR;
		f = f == Z_FINISH ? Z_BUF_ERROR : Z_OK;
		r = Z_BUF_ERROR;
		for (;;) switch (z->state->mode) {
			case IM_METHOD:
				IM_NEEDBYTE
					if (((z->state->sub.method = IM_NEXTBYTE) & 0xf) != Z_DEFLATED) {
						z->state->mode = IM_BAD;
						z->msg = (char*)"unknown compression method";
						z->state->sub.marker = 5;       // can't try inflateSync
						break;
					}
				if ((z->state->sub.method >> 4) + 8 > z->state->wbits) {
					z->state->mode = IM_BAD;
					z->msg = (char*)"invalid window size";
					z->state->sub.marker = 5;       // can't try inflateSync
					break;
				}
				z->state->mode = IM_FLAG;
			case IM_FLAG:
				IM_NEEDBYTE
					b = IM_NEXTBYTE;
				if (((z->state->sub.method << 8) + b) % 31) {
					z->state->mode = IM_BAD;
					z->msg = (char*)"incorrect header check";
					z->state->sub.marker = 5;       // can't try inflateSync 
					break;
				}
				LuTracev((stderr, "inflate: zlib header ok\n"));
				if (!(b & PRESET_DICT)) {
					z->state->mode = IM_BLOCKS;
					break;
				}
				z->state->mode = IM_DICT4;
			case IM_DICT4:
				IM_NEEDBYTE
					z->state->sub.check.need = (uLong)IM_NEXTBYTE << 24;
				z->state->mode = IM_DICT3;
			case IM_DICT3:
				IM_NEEDBYTE
					z->state->sub.check.need += (uLong)IM_NEXTBYTE << 16;
				z->state->mode = IM_DICT2;
			case IM_DICT2:
				IM_NEEDBYTE
					z->state->sub.check.need += (uLong)IM_NEXTBYTE << 8;
				z->state->mode = IM_DICT1;
			case IM_DICT1:
				IM_NEEDBYTE; r;
				z->state->sub.check.need += (uLong)IM_NEXTBYTE;
				z->adler = z->state->sub.check.need;
				z->state->mode = IM_DICT0;
				return Z_NEED_DICT;
			case IM_DICT0:
				z->state->mode = IM_BAD;
				z->msg = (char*)"need dictionary";
				z->state->sub.marker = 0;       // can try inflateSync 
				return Z_STREAM_ERROR;
			case IM_BLOCKS:
				r = inflate_blocks(z->state->blocks, z, r);
				if (r == Z_DATA_ERROR) {
					z->state->mode = IM_BAD;
					z->state->sub.marker = 0;       // can try inflateSync 
					break;
				}
				if (r == Z_OK)
					r = f;
				if (r != Z_STREAM_END)
					return r;
				r = f;
				inflate_blocks_reset(z->state->blocks, z, &z->state->sub.check.was);
				if (z->state->nowrap) {
					z->state->mode = IM_DONE;
					break;
				}
				z->state->mode = IM_CHECK4;
			case IM_CHECK4:
				IM_NEEDBYTE
					z->state->sub.check.need = (uLong)IM_NEXTBYTE << 24;
				z->state->mode = IM_CHECK3;
			case IM_CHECK3:
				IM_NEEDBYTE
					z->state->sub.check.need += (uLong)IM_NEXTBYTE << 16;
				z->state->mode = IM_CHECK2;
			case IM_CHECK2:
				IM_NEEDBYTE
					z->state->sub.check.need += (uLong)IM_NEXTBYTE << 8;
				z->state->mode = IM_CHECK1;
			case IM_CHECK1:
				IM_NEEDBYTE
					z->state->sub.check.need += (uLong)IM_NEXTBYTE;

				if (z->state->sub.check.was != z->state->sub.check.need) {
					z->state->mode = IM_BAD;
					z->msg = (char*)"incorrect data check";
					z->state->sub.marker = 5;       // can't try inflateSync 
					break;
				}
				LuTracev((stderr, "inflate: zlib check ok\n"));
				z->state->mode = IM_DONE;
			case IM_DONE:
				return Z_STREAM_END;
			case IM_BAD:
				return Z_DATA_ERROR;
			default:
				return Z_STREAM_ERROR;
		}
	}





	// unzip.c -- IO on .zip files using zlib
	// Version 0.15 beta, Mar 19th, 1998,
	// Read unzip.h for more info




#define UNZ_BUFSIZE (16384)
#define UNZ_MAXFILENAMEINZIP (256)
#define SIZECENTRALDIRITEM (0x2e)
#define SIZEZIPLOCALHEADER (0x1e)




	const char unz_copyright[] = " unzip 0.15 Copyright 1998 Gilles Vollant ";

	// unz_file_info_interntal contain internal info about a file in zipfile
	typedef struct unz_file_info_internal_s {
		uLong offset_curfile;// relative offset of local header 4 bytes
	} unz_file_info_internal;


	typedef struct {
		bool is_handle; // either a handle or memory
		bool canseek;
		// for handles:
		HANDLE h; bool herr; unsigned long initial_offset; bool mustclosehandle;
		// for memory:
		void *buf; unsigned int len, pos; // if it's a memory block
	} LUFILE;


	inline LUFILE *lufopen(void *z, unsigned int len, DWORD flags, ZRESULT *err) {
		if (flags != ZIP_HANDLE && flags != ZIP_FILENAME && flags != ZIP_MEMORY) {
			*err = ZR_ARGS; return NULL;
		}
		//
		HANDLE h = 0; bool canseek = false; *err = ZR_OK;
		bool mustclosehandle = false;
		if (flags == ZIP_HANDLE || flags == ZIP_FILENAME) {
			if (flags == ZIP_HANDLE) {
				HANDLE hf = (HANDLE)z;
				h = hf; mustclosehandle = false;
#ifdef DuplicateHandle
				BOOL res = DuplicateHandle(GetCurrentProcess(), hf, GetCurrentProcess(), &h, 0, FALSE, DUPLICATE_SAME_ACCESS);
				if (!res) mustclosehandle = true;
#endif
			} else {

				h = CreateFile((const TCHAR*)z, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
				if (h == INVALID_HANDLE_VALUE) {
					*err = ZR_NOFILE; return NULL;
				}
				mustclosehandle = true;
			}
			// test if we can seek on it. We can't use GetFileType(h)==FILE_TYPE_DISK since it's not on CE.
			DWORD res = GetFilePosU(h);
			canseek = (res != 0xFFFFFFFF);
		}
		LUFILE *lf = new LUFILE;
		if (flags == ZIP_HANDLE || flags == ZIP_FILENAME) {
			lf->is_handle = true; lf->mustclosehandle = mustclosehandle;
			lf->canseek = canseek;
			lf->h = h; lf->herr = false;
			lf->initial_offset = 0;
			if (canseek) lf->initial_offset = GetFilePosU(h);
		} else {
			lf->is_handle = false;
			lf->canseek = true;
			lf->mustclosehandle = false;
			lf->buf = z; lf->len = len; lf->pos = 0; lf->initial_offset = 0;
		}
		*err = ZR_OK;
		return lf;
	}


	inline int lufclose(LUFILE *stream) {
		if (stream == NULL) return EOF;

		if (stream->mustclosehandle) CloseHandle(stream->h);
		delete stream;
		return 0;
	}

	inline int luferror(LUFILE *stream) {
		if (stream->is_handle && stream->herr) return 1;
		else return 0;
	}

	inline long int luftell(LUFILE *stream) {
		if (stream->is_handle && stream->canseek) return GetFilePosU(stream->h) - stream->initial_offset;
		else if (stream->is_handle) return 0;
		else return stream->pos;
	}

	inline int lufseek(LUFILE *stream, long offset, int whence) {
		if (stream->is_handle && stream->canseek) {

			if (whence == SEEK_SET) SetFilePointer(stream->h, stream->initial_offset + offset, 0, FILE_BEGIN);
			else if (whence == SEEK_CUR) SetFilePointer(stream->h, offset, NULL, FILE_CURRENT);
			else if (whence == SEEK_END) SetFilePointer(stream->h, offset, NULL, FILE_END);
			else return 19; // EINVAL
			return 0;
		} else if (stream->is_handle) return 29; // ESPIPE
		else {
			if (whence == SEEK_SET) stream->pos = offset;
			else if (whence == SEEK_CUR) stream->pos += offset;
			else if (whence == SEEK_END) stream->pos = stream->len + offset;
			return 0;
		}
	}


	inline size_t lufread(void *ptr, size_t size, size_t n, LUFILE *stream) {
		unsigned int toread = (unsigned int)(size*n);
		if (stream->is_handle) {

			DWORD red; BOOL res = ReadFile(stream->h, ptr, toread, &red, NULL);
			if (!res) stream->herr = true;
			return red / size;
		}
		if (stream->pos + toread > stream->len) toread = stream->len - stream->pos;
		memcpy(ptr, (char*)stream->buf + stream->pos, toread); DWORD red = toread;
		stream->pos += red;
		return red / size;
	}




	// file_in_zip_read_info_s contain internal information about a file in zipfile,
	//  when reading and decompress it
	typedef struct {
		char  *read_buffer;         // internal buffer for compressed data
		z_stream stream;            // zLib stream structure for inflate

		uLong pos_in_zipfile;       // position in byte on the zipfile, for fseek
		uLong stream_initialised;   // flag set if stream structure is initialised

		uLong offset_local_extrafield;// offset of the local extra field
		uInt  size_local_extrafield;// size of the local extra field
		uLong pos_local_extrafield;   // position in the local extra field in read

		uLong crc32;                // crc32 of all data uncompressed
		uLong crc32_wait;           // crc32 we must obtain after decompress all
		uLong rest_read_compressed; // number of byte to be decompressed
		uLong rest_read_uncompressed;//number of byte to be obtained after decomp
		LUFILE* file;                 // io structore of the zipfile
		uLong compression_method;   // compression method (0==store)
		uLong byte_before_the_zipfile;// byte before the zipfile, (>0 for sfx)
		bool encrypted;               // is it encrypted?
		unsigned long keys[3];        // decryption keys, initialized by unzOpenCurrentFile
		int encheadleft;              // the first call(s) to unzReadCurrentFile will read this many encryption-header bytes first
		char crcenctest;              // if encrypted, we'll check the encryption buffer against this
	} file_in_zip_read_info_s;


	// unz_s contain internal information about the zipfile
	typedef struct {
		LUFILE* file;               // io structore of the zipfile
		unz_global_info gi;         // public global information
		uLong byte_before_the_zipfile;// byte before the zipfile, (>0 for sfx)
		uLong num_file;             // number of the current file in the zipfile
		uLong pos_in_central_dir;   // pos of the current file in the central dir
		uLong current_file_ok;      // flag about the usability of the current file
		uLong central_pos;          // position of the beginning of the central dir

		uLong size_central_dir;     // size of the central directory
		uLong offset_central_dir;   // offset of start of central directory with respect to the starting disk number

		unz_file_info cur_file_info; // public info about the current file in zip
		unz_file_info_internal cur_file_info_internal; // private info about it
		file_in_zip_read_info_s* pfile_in_zip_read; // structure about the current file if we are decompressing it
	} unz_s, *unzFile;


	inline int unzStringFileNameCompare(const char* fileName1, const char* fileName2, int iCaseSensitivity);
	//   Compare two filename (fileName1,fileName2).

	inline z_off_t unztell(unzFile file);
	//  Give the current position in uncompressed data

	inline int unzeof(unzFile file);
	//  return 1 if the end of file was reached, 0 elsewhere

	inline int unzGetLocalExtrafield(unzFile file, voidp buf, unsigned len);
	//  Read extra field from the current file (opened by unzOpenCurrentFile)
	//  This is the local-header version of the extra field (sometimes, there is
	//    more info in the local-header version than in the central-header)
	//
	//  if buf==NULL, it return the size of the local extra field
	//
	//  if buf!=NULL, len is the size of the buffer, the extra header is copied in
	//	buf.
	//  the return value is the number of bytes copied in buf, or (if <0)
	//	the error code



	// ===========================================================================
	//   Read a byte from a gz_stream; update next_in and avail_in. Return EOF
	// for end of file.
	// IN assertion: the stream s has been sucessfully opened for reading.

	inline int unzlocal_getByte(LUFILE *fin, int *pi) {
		unsigned char c;
		int err = (int)lufread(&c, 1, 1, fin);
		if (err == 1) {
			*pi = (int)c;
			return UNZ_OK;
		} else {
			if (luferror(fin)) return UNZ_ERRNO;
			else return UNZ_EOF;
		}
	}


	// ===========================================================================
	// Reads a long in LSB order from the given gz_stream. Sets
	inline int unzlocal_getShort(LUFILE *fin, uLong *pX) {
		uLong x;
		int i;
		int err;

		err = unzlocal_getByte(fin, &i);
		x = (uLong)i;

		if (err == UNZ_OK)
			err = unzlocal_getByte(fin, &i);
		x += ((uLong)i) << 8;

		if (err == UNZ_OK)
			*pX = x;
		else
			*pX = 0;
		return err;
	}

	inline int unzlocal_getLong(LUFILE *fin, uLong *pX) {
		uLong x;
		int i;
		int err;

		err = unzlocal_getByte(fin, &i);
		x = (uLong)i;

		if (err == UNZ_OK)
			err = unzlocal_getByte(fin, &i);
		x += ((uLong)i) << 8;

		if (err == UNZ_OK)
			err = unzlocal_getByte(fin, &i);
		x += ((uLong)i) << 16;

		if (err == UNZ_OK)
			err = unzlocal_getByte(fin, &i);
		x += ((uLong)i) << 24;

		if (err == UNZ_OK)
			*pX = x;
		else
			*pX = 0;
		return err;
	}


	// My own strcmpi / strcasecmp 
	inline int strcmpcasenosensitive_internal(const char* fileName1, const char *fileName2) {
		for (;;) {
			char c1 = *(fileName1++);
			char c2 = *(fileName2++);
			if ((c1 >= 'a') && (c1 <= 'z'))
				c1 -= (char)0x20;
			if ((c2 >= 'a') && (c2 <= 'z'))
				c2 -= (char)0x20;
			if (c1 == '\0')
				return ((c2 == '\0') ? 0 : -1);
			if (c2 == '\0')
				return 1;
			if (c1 < c2)
				return -1;
			if (c1 > c2)
				return 1;
		}
	}




	//
	// Compare two filename (fileName1,fileName2).
	// If iCaseSenisivity = 1, comparision is case sensitivity (like strcmp)
	// If iCaseSenisivity = 2, comparision is not case sensitivity (like strcmpi or strcasecmp)
	//
	inline int unzStringFileNameCompare(const char*fileName1, const char*fileName2, int iCaseSensitivity) {
		if (iCaseSensitivity == 1) return strcmp(fileName1, fileName2);
		else return strcmpcasenosensitive_internal(fileName1, fileName2);
	}

#define BUFREADCOMMENT (0x400)


	//  Locate the Central directory of a zipfile (at the end, just before
	// the global comment). Lu bugfix 2005.07.26 - returns 0xFFFFFFFF if not found,
	// rather than 0, since 0 is a valid central-dir-location for an empty zipfile.
	inline uLong unzlocal_SearchCentralDir(LUFILE *fin) {
		if (lufseek(fin, 0, SEEK_END) != 0) return 0xFFFFFFFF;
		uLong uSizeFile = luftell(fin);

		uLong uMaxBack = 0xffff; // maximum size of global comment
		if (uMaxBack > uSizeFile) uMaxBack = uSizeFile;

		unsigned char *buf = (unsigned char*)zmalloc(BUFREADCOMMENT + 4);
		if (buf == NULL) return 0xFFFFFFFF;
		uLong uPosFound = 0xFFFFFFFF;

		uLong uBackRead = 4;
		while (uBackRead < uMaxBack) {
			uLong uReadSize, uReadPos;
			int i;
			if (uBackRead + BUFREADCOMMENT > uMaxBack) uBackRead = uMaxBack;
			else uBackRead += BUFREADCOMMENT;
			uReadPos = uSizeFile - uBackRead;
			uReadSize = ((BUFREADCOMMENT + 4) < (uSizeFile - uReadPos)) ? (BUFREADCOMMENT + 4) : (uSizeFile - uReadPos);
			if (lufseek(fin, uReadPos, SEEK_SET) != 0) break;
			if (lufread(buf, (uInt)uReadSize, 1, fin) != 1) break;
			for (i = (int)uReadSize - 3; (i--) >= 0;) {
				if (((*(buf + i)) == 0x50) && ((*(buf + i + 1)) == 0x4b) && ((*(buf + i + 2)) == 0x05) && ((*(buf + i + 3)) == 0x06)) {
					uPosFound = uReadPos + i;	break;
				}
			}
			if (uPosFound != 0) break;
		}
		if (buf) zfree(buf);
		return uPosFound;
	}


	inline int unzGoToFirstFile(unzFile file);
	inline int unzCloseCurrentFile(unzFile file);

	// Open a Zip file.
	// If the zipfile cannot be opened (file don't exist or in not valid), return NULL.
	// Otherwise, the return value is a unzFile Handle, usable with other unzip functions
	inline unzFile unzOpenInternal(LUFILE *fin) {
		if (fin == NULL) return NULL;
		if (unz_copyright[0] != ' ') {
			lufclose(fin); return NULL;
		}

		int err = UNZ_OK;
		unz_s us = { 0 };
		uLong central_pos = 0, uL = 0;
		central_pos = unzlocal_SearchCentralDir(fin);
		if (central_pos == 0xFFFFFFFF) err = UNZ_ERRNO;
		if (err == UNZ_OK && lufseek(fin, central_pos, SEEK_SET) != 0) err = UNZ_ERRNO;
		// the signature, already checked
		if (err == UNZ_OK && unzlocal_getLong(fin, &uL) != UNZ_OK) err = UNZ_ERRNO;
		// number of this disk
		uLong number_disk = 0;          // number of the current dist, used for spanning ZIP, unsupported, always 0
		if (err == UNZ_OK && unzlocal_getShort(fin, &number_disk) != UNZ_OK) err = UNZ_ERRNO;
		// number of the disk with the start of the central directory
		uLong number_disk_with_CD = 0;  // number the the disk with central dir, used for spaning ZIP, unsupported, always 0
		if (err == UNZ_OK && unzlocal_getShort(fin, &number_disk_with_CD) != UNZ_OK) err = UNZ_ERRNO;
		// total number of entries in the central dir on this disk
		if (err == UNZ_OK && unzlocal_getShort(fin, &us.gi.number_entry) != UNZ_OK) err = UNZ_ERRNO;
		// total number of entries in the central dir
		uLong number_entry_CD = 0;      // total number of entries in the central dir (same than number_entry on nospan)
		if (err == UNZ_OK && unzlocal_getShort(fin, &number_entry_CD) != UNZ_OK) err = UNZ_ERRNO;
		if (err == UNZ_OK && ((number_entry_CD != us.gi.number_entry) || (number_disk_with_CD != 0) || (number_disk != 0))) err = UNZ_BADZIPFILE;
		// size of the central directory
		if (err == UNZ_OK && unzlocal_getLong(fin, &us.size_central_dir) != UNZ_OK) err = UNZ_ERRNO;
		// offset of start of central directory with respect to the starting disk number
		if (err == UNZ_OK && unzlocal_getLong(fin, &us.offset_central_dir) != UNZ_OK) err = UNZ_ERRNO;
		// zipfile comment length
		if (err == UNZ_OK && unzlocal_getShort(fin, &us.gi.size_comment) != UNZ_OK) err = UNZ_ERRNO;
		if (err == UNZ_OK && ((central_pos + fin->initial_offset < us.offset_central_dir + us.size_central_dir) && (err == UNZ_OK))) err = UNZ_BADZIPFILE;
		if (err != UNZ_OK) {
			lufclose(fin); return NULL;
		}

		us.file = fin;
		us.byte_before_the_zipfile = central_pos + fin->initial_offset - (us.offset_central_dir + us.size_central_dir);
		us.central_pos = central_pos;
		us.pfile_in_zip_read = NULL;
		fin->initial_offset = 0; // since the zipfile itself is expected to handle this

		unz_s *s = (unz_s*)zmalloc(sizeof(unz_s));
		*s = us;
		unzGoToFirstFile((unzFile)s);
		return (unzFile)s;
	}



	//  Close a ZipFile opened with unzipOpen.
	//  If there is files inside the .Zip opened with unzipOpenCurrentFile (see later),
	//    these files MUST be closed with unzipCloseCurrentFile before call unzipClose.
	//  return UNZ_OK if there is no problem.
	inline int unzClose(unzFile file) {
		unz_s* s;
		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;

		if (s->pfile_in_zip_read != NULL)
			unzCloseCurrentFile(file);

		lufclose(s->file);
		if (s) zfree(s); // unused s=0;
		return UNZ_OK;
	}


	//  Write info about the ZipFile in the *pglobal_info structure.
	//  No preparation of the structure is needed
	//  return UNZ_OK if there is no problem. 
	inline int unzGetGlobalInfo(unzFile file, unz_global_info *pglobal_info) {
		unz_s* s;
		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		*pglobal_info = s->gi;
		return UNZ_OK;
	}


	//   Translate date/time from Dos format to tm_unz (readable more easilty)
	inline void unzlocal_DosDateToTmuDate(uLong ulDosDate, tm_unz* ptm) {
		uLong uDate;
		uDate = (uLong)(ulDosDate >> 16);
		ptm->tm_mday = (uInt)(uDate & 0x1f);
		ptm->tm_mon = (uInt)((((uDate) & 0x1E0) / 0x20) - 1);
		ptm->tm_year = (uInt)(((uDate & 0x0FE00) / 0x0200) + 1980);

		ptm->tm_hour = (uInt)((ulDosDate & 0xF800) / 0x800);
		ptm->tm_min = (uInt)((ulDosDate & 0x7E0) / 0x20);
		ptm->tm_sec = (uInt)(2 * (ulDosDate & 0x1f));
	}

	//  Get Info about the current file in the zipfile, with internal only info
	inline int unzlocal_GetCurrentFileInfoInternal(unzFile file,
												   unz_file_info *pfile_info,
												   unz_file_info_internal
												   *pfile_info_internal,
												   char *szFileName,
												   uLong fileNameBufferSize,
												   void *extraField,
												   uLong extraFieldBufferSize,
												   char *szComment,
												   uLong commentBufferSize);

	inline int unzlocal_GetCurrentFileInfoInternal(unzFile file, unz_file_info *pfile_info,
												   unz_file_info_internal *pfile_info_internal, char *szFileName,
												   uLong fileNameBufferSize, void *extraField, uLong extraFieldBufferSize,
												   char *szComment, uLong commentBufferSize) {
		unz_s* s;
		unz_file_info file_info;
		unz_file_info_internal file_info_internal;
		int err = UNZ_OK;
		uLong uMagic;
		long lSeek = 0;

		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		if (lufseek(s->file, s->pos_in_central_dir + s->byte_before_the_zipfile, SEEK_SET) != 0)
			err = UNZ_ERRNO;


		// we check the magic
		if (err == UNZ_OK)
			if (unzlocal_getLong(s->file, &uMagic) != UNZ_OK)
				err = UNZ_ERRNO;
			else if (uMagic != 0x02014b50)
				err = UNZ_BADZIPFILE;

		if (unzlocal_getShort(s->file, &file_info.version) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.version_needed) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.flag) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.compression_method) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getLong(s->file, &file_info.dosDate) != UNZ_OK)
			err = UNZ_ERRNO;

		unzlocal_DosDateToTmuDate(file_info.dosDate, &file_info.tmu_date);

		if (unzlocal_getLong(s->file, &file_info.crc) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getLong(s->file, &file_info.compressed_size) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getLong(s->file, &file_info.uncompressed_size) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.size_filename) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.size_file_extra) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.size_file_comment) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.disk_num_start) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &file_info.internal_fa) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getLong(s->file, &file_info.external_fa) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getLong(s->file, &file_info_internal.offset_curfile) != UNZ_OK)
			err = UNZ_ERRNO;

		lSeek += file_info.size_filename;
		if ((err == UNZ_OK) && (szFileName != NULL)) {
			uLong uSizeRead;
			if (file_info.size_filename < fileNameBufferSize) {
				*(szFileName + file_info.size_filename) = '\0';
				uSizeRead = file_info.size_filename;
			} else
				uSizeRead = fileNameBufferSize;

			if ((file_info.size_filename > 0) && (fileNameBufferSize > 0))
				if (lufread(szFileName, (uInt)uSizeRead, 1, s->file) != 1)
					err = UNZ_ERRNO;
			lSeek -= uSizeRead;
		}


		if ((err == UNZ_OK) && (extraField != NULL)) {
			uLong uSizeRead;
			if (file_info.size_file_extra < extraFieldBufferSize)
				uSizeRead = file_info.size_file_extra;
			else
				uSizeRead = extraFieldBufferSize;

			if (lSeek != 0)
				if (lufseek(s->file, lSeek, SEEK_CUR) == 0)
					lSeek = 0;
				else
					err = UNZ_ERRNO;
			if ((file_info.size_file_extra > 0) && (extraFieldBufferSize > 0))
				if (lufread(extraField, (uInt)uSizeRead, 1, s->file) != 1)
					err = UNZ_ERRNO;
			lSeek += file_info.size_file_extra - uSizeRead;
		} else
			lSeek += file_info.size_file_extra;


		if ((err == UNZ_OK) && (szComment != NULL)) {
			uLong uSizeRead;
			if (file_info.size_file_comment < commentBufferSize) {
				*(szComment + file_info.size_file_comment) = '\0';
				uSizeRead = file_info.size_file_comment;
			} else
				uSizeRead = commentBufferSize;

			if (lSeek != 0)
				if (lufseek(s->file, lSeek, SEEK_CUR) == 0) {
				} // unused lSeek=0;
				else
					err = UNZ_ERRNO;
			if ((file_info.size_file_comment > 0) && (commentBufferSize > 0))
				if (lufread(szComment, (uInt)uSizeRead, 1, s->file) != 1)
					err = UNZ_ERRNO;
			//unused lSeek+=file_info.size_file_comment - uSizeRead;
		} else {
		} //unused lSeek+=file_info.size_file_comment;

		if ((err == UNZ_OK) && (pfile_info != NULL))
			*pfile_info = file_info;

		if ((err == UNZ_OK) && (pfile_info_internal != NULL))
			*pfile_info_internal = file_info_internal;

		return err;
	}



	//  Write info about the ZipFile in the *pglobal_info structure.
	//  No preparation of the structure is needed
	//  return UNZ_OK if there is no problem.
	inline int unzGetCurrentFileInfo(unzFile file, unz_file_info *pfile_info,
									 char *szFileName, uLong fileNameBufferSize, void *extraField, uLong extraFieldBufferSize,
									 char *szComment, uLong commentBufferSize) {
		return unzlocal_GetCurrentFileInfoInternal(file, pfile_info, NULL, szFileName, fileNameBufferSize,
												   extraField, extraFieldBufferSize, szComment, commentBufferSize);
	}


	//  Set the current file of the zipfile to the first file.
	//  return UNZ_OK if there is no problem
	inline int unzGoToFirstFile(unzFile file) {
		int err;
		unz_s* s;
		if (file == NULL) return UNZ_PARAMERROR;
		s = (unz_s*)file;
		s->pos_in_central_dir = s->offset_central_dir;
		s->num_file = 0;
		err = unzlocal_GetCurrentFileInfoInternal(file, &s->cur_file_info,
												  &s->cur_file_info_internal,
												  NULL, 0, NULL, 0, NULL, 0);
		s->current_file_ok = (err == UNZ_OK);
		return err;
	}


	//  Set the current file of the zipfile to the next file.
	//  return UNZ_OK if there is no problem
	//  return UNZ_END_OF_LIST_OF_FILE if the actual file was the latest.
	inline int unzGoToNextFile(unzFile file) {
		unz_s* s;
		int err;

		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		if (!s->current_file_ok)
			return UNZ_END_OF_LIST_OF_FILE;
		if (s->num_file + 1 == s->gi.number_entry)
			return UNZ_END_OF_LIST_OF_FILE;

		s->pos_in_central_dir += SIZECENTRALDIRITEM + s->cur_file_info.size_filename +
			s->cur_file_info.size_file_extra + s->cur_file_info.size_file_comment;
		s->num_file++;
		err = unzlocal_GetCurrentFileInfoInternal(file, &s->cur_file_info,
												  &s->cur_file_info_internal,
												  NULL, 0, NULL, 0, NULL, 0);
		s->current_file_ok = (err == UNZ_OK);
		return err;
	}


	//  Try locate the file szFileName in the zipfile.
	//  For the iCaseSensitivity signification, see unzStringFileNameCompare
	//  return value :
	//  UNZ_OK if the file is found. It becomes the current file.
	//  UNZ_END_OF_LIST_OF_FILE if the file is not found
	inline int unzLocateFile(unzFile file, const char *szFileName, int iCaseSensitivity) {
		unz_s* s;
		int err;


		uLong num_fileSaved;
		uLong pos_in_central_dirSaved;


		if (file == NULL)
			return UNZ_PARAMERROR;

		if (strlen(szFileName) >= UNZ_MAXFILENAMEINZIP)
			return UNZ_PARAMERROR;

		s = (unz_s*)file;
		if (!s->current_file_ok)
			return UNZ_END_OF_LIST_OF_FILE;

		num_fileSaved = s->num_file;
		pos_in_central_dirSaved = s->pos_in_central_dir;

		err = unzGoToFirstFile(file);

		while (err == UNZ_OK) {
			char szCurrentFileName[UNZ_MAXFILENAMEINZIP + 1];
			unzGetCurrentFileInfo(file, NULL,
								  szCurrentFileName, sizeof(szCurrentFileName) - 1,
								  NULL, 0, NULL, 0);
			if (unzStringFileNameCompare(szCurrentFileName, szFileName, iCaseSensitivity) == 0)
				return UNZ_OK;
			err = unzGoToNextFile(file);
		}

		s->num_file = num_fileSaved;
		s->pos_in_central_dir = pos_in_central_dirSaved;
		return err;
	}


	//  Read the local header of the current zipfile
	//  Check the coherency of the local header and info in the end of central
	//        directory about this file
	//  store in *piSizeVar the size of extra info in local header
	//        (filename and size of extra field data)
	inline int unzlocal_CheckCurrentFileCoherencyHeader(unz_s *s, uInt *piSizeVar,
														uLong *poffset_local_extrafield, uInt  *psize_local_extrafield) {
		uLong uMagic, uData, uFlags;
		uLong size_filename;
		uLong size_extra_field;
		int err = UNZ_OK;

		*piSizeVar = 0;
		*poffset_local_extrafield = 0;
		*psize_local_extrafield = 0;

		if (lufseek(s->file, s->cur_file_info_internal.offset_curfile + s->byte_before_the_zipfile, SEEK_SET) != 0)
			return UNZ_ERRNO;


		if (err == UNZ_OK)
			if (unzlocal_getLong(s->file, &uMagic) != UNZ_OK)
				err = UNZ_ERRNO;
			else if (uMagic != 0x04034b50)
				err = UNZ_BADZIPFILE;

		if (unzlocal_getShort(s->file, &uData) != UNZ_OK)
			err = UNZ_ERRNO;
		//	else if ((err==UNZ_OK) && (uData!=s->cur_file_info.wVersion))
		//		err=UNZ_BADZIPFILE;
		if (unzlocal_getShort(s->file, &uFlags) != UNZ_OK)
			err = UNZ_ERRNO;

		if (unzlocal_getShort(s->file, &uData) != UNZ_OK)
			err = UNZ_ERRNO;
		else if ((err == UNZ_OK) && (uData != s->cur_file_info.compression_method))
			err = UNZ_BADZIPFILE;

		if ((err == UNZ_OK) && (s->cur_file_info.compression_method != 0) &&
			(s->cur_file_info.compression_method != Z_DEFLATED))
			err = UNZ_BADZIPFILE;

		if (unzlocal_getLong(s->file, &uData) != UNZ_OK) // date/time
			err = UNZ_ERRNO;

		if (unzlocal_getLong(s->file, &uData) != UNZ_OK) // crc
			err = UNZ_ERRNO;
		else if ((err == UNZ_OK) && (uData != s->cur_file_info.crc) &&
			((uFlags & 8) == 0))
			err = UNZ_BADZIPFILE;

		if (unzlocal_getLong(s->file, &uData) != UNZ_OK) // size compr
			err = UNZ_ERRNO;
		else if ((err == UNZ_OK) && (uData != s->cur_file_info.compressed_size) &&
			((uFlags & 8) == 0))
			err = UNZ_BADZIPFILE;

		if (unzlocal_getLong(s->file, &uData) != UNZ_OK) // size uncompr
			err = UNZ_ERRNO;
		else if ((err == UNZ_OK) && (uData != s->cur_file_info.uncompressed_size) &&
			((uFlags & 8) == 0))
			err = UNZ_BADZIPFILE;


		if (unzlocal_getShort(s->file, &size_filename) != UNZ_OK)
			err = UNZ_ERRNO;
		else if ((err == UNZ_OK) && (size_filename != s->cur_file_info.size_filename))
			err = UNZ_BADZIPFILE;

		*piSizeVar += (uInt)size_filename;

		if (unzlocal_getShort(s->file, &size_extra_field) != UNZ_OK)
			err = UNZ_ERRNO;
		*poffset_local_extrafield = s->cur_file_info_internal.offset_curfile +
			SIZEZIPLOCALHEADER + size_filename;
		*psize_local_extrafield = (uInt)size_extra_field;

		*piSizeVar += (uInt)size_extra_field;

		return err;
	}





	//  Open for reading data the current file in the zipfile.
	//  If there is no error and the file is opened, the return value is UNZ_OK.
	inline int unzOpenCurrentFile(unzFile file, const char *password) {
		int err;
		int Store;
		uInt iSizeVar;
		unz_s* s;
		file_in_zip_read_info_s* pfile_in_zip_read_info;
		uLong offset_local_extrafield;  // offset of the local extra field
		uInt  size_local_extrafield;    // size of the local extra field

		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		if (!s->current_file_ok)
			return UNZ_PARAMERROR;

		if (s->pfile_in_zip_read != NULL)
			unzCloseCurrentFile(file);

		if (unzlocal_CheckCurrentFileCoherencyHeader(s, &iSizeVar,
													 &offset_local_extrafield, &size_local_extrafield) != UNZ_OK)
			return UNZ_BADZIPFILE;

		pfile_in_zip_read_info = (file_in_zip_read_info_s*)zmalloc(sizeof(file_in_zip_read_info_s));
		if (pfile_in_zip_read_info == NULL)
			return UNZ_INTERNALERROR;

		pfile_in_zip_read_info->read_buffer = (char*)zmalloc(UNZ_BUFSIZE);
		pfile_in_zip_read_info->offset_local_extrafield = offset_local_extrafield;
		pfile_in_zip_read_info->size_local_extrafield = size_local_extrafield;
		pfile_in_zip_read_info->pos_local_extrafield = 0;

		if (pfile_in_zip_read_info->read_buffer == NULL) {
			if (pfile_in_zip_read_info != 0) zfree(pfile_in_zip_read_info); //unused pfile_in_zip_read_info=0;
			return UNZ_INTERNALERROR;
		}

		pfile_in_zip_read_info->stream_initialised = 0;

		if ((s->cur_file_info.compression_method != 0) && (s->cur_file_info.compression_method != Z_DEFLATED)) { // unused err=UNZ_BADZIPFILE;
		}
		Store = s->cur_file_info.compression_method == 0;

		pfile_in_zip_read_info->crc32_wait = s->cur_file_info.crc;
		pfile_in_zip_read_info->crc32 = 0;
		pfile_in_zip_read_info->compression_method = s->cur_file_info.compression_method;
		pfile_in_zip_read_info->file = s->file;
		pfile_in_zip_read_info->byte_before_the_zipfile = s->byte_before_the_zipfile;

		pfile_in_zip_read_info->stream.total_out = 0;

		if (!Store) {
			pfile_in_zip_read_info->stream.zalloc = (alloc_func)0;
			pfile_in_zip_read_info->stream.zfree = (free_func)0;
			pfile_in_zip_read_info->stream.opaque = (voidpf)0;

			err = inflateInit2(&pfile_in_zip_read_info->stream);
			if (err == Z_OK)
				pfile_in_zip_read_info->stream_initialised = 1;
			// windowBits is passed < 0 to tell that there is no zlib header.
			// Note that in this case inflate *requires* an extra "dummy" byte
			// after the compressed stream in order to complete decompression and
			// return Z_STREAM_END.
			// In unzip, i don't wait absolutely Z_STREAM_END because I known the
			// size of both compressed and uncompressed data
		}
		pfile_in_zip_read_info->rest_read_compressed = s->cur_file_info.compressed_size;
		pfile_in_zip_read_info->rest_read_uncompressed = s->cur_file_info.uncompressed_size;
		pfile_in_zip_read_info->encrypted = (s->cur_file_info.flag & 1) != 0;
		bool extlochead = (s->cur_file_info.flag & 8) != 0;
		if (extlochead) pfile_in_zip_read_info->crcenctest = (char)((s->cur_file_info.dosDate >> 8) & 0xff);
		else pfile_in_zip_read_info->crcenctest = (char)(s->cur_file_info.crc >> 24);
		pfile_in_zip_read_info->encheadleft = (pfile_in_zip_read_info->encrypted ? 12 : 0);
		pfile_in_zip_read_info->keys[0] = 305419896L;
		pfile_in_zip_read_info->keys[1] = 591751049L;
		pfile_in_zip_read_info->keys[2] = 878082192L;
		for (const char *cp = password; cp != 0 && *cp != 0; cp++) Uupdate_keys(pfile_in_zip_read_info->keys, *cp);

		pfile_in_zip_read_info->pos_in_zipfile =
			s->cur_file_info_internal.offset_curfile + SIZEZIPLOCALHEADER +
			iSizeVar;

		pfile_in_zip_read_info->stream.avail_in = (uInt)0;

		s->pfile_in_zip_read = pfile_in_zip_read_info;

		return UNZ_OK;
	}


	//  Read bytes from the current file.
	//  buf contain buffer where data must be copied
	//  len the size of buf.
	//  return the number of byte copied if somes bytes are copied (and also sets *reached_eof)
	//  return 0 if the end of file was reached. (and also sets *reached_eof).
	//  return <0 with error code if there is an error. (in which case *reached_eof is meaningless)
	//    (UNZ_ERRNO for IO error, or zLib error for uncompress error)
	inline int unzReadCurrentFile(unzFile file, voidp buf, unsigned len, bool *reached_eof) {
		int err = UNZ_OK;
		uInt iRead = 0;
		if (reached_eof != 0) *reached_eof = false;

		unz_s *s = (unz_s*)file;
		if (s == NULL) return UNZ_PARAMERROR;

		file_in_zip_read_info_s* pfile_in_zip_read_info = s->pfile_in_zip_read;
		if (pfile_in_zip_read_info == NULL) return UNZ_PARAMERROR;
		if ((pfile_in_zip_read_info->read_buffer == NULL)) return UNZ_END_OF_LIST_OF_FILE;
		if (len == 0) return 0;

		pfile_in_zip_read_info->stream.next_out = (Byte*)buf;
		pfile_in_zip_read_info->stream.avail_out = (uInt)len;

		if (len > pfile_in_zip_read_info->rest_read_uncompressed) {
			pfile_in_zip_read_info->stream.avail_out = (uInt)pfile_in_zip_read_info->rest_read_uncompressed;
		}

		while (pfile_in_zip_read_info->stream.avail_out > 0) {
			if ((pfile_in_zip_read_info->stream.avail_in == 0) && (pfile_in_zip_read_info->rest_read_compressed > 0)) {
				uInt uReadThis = UNZ_BUFSIZE;
				if (pfile_in_zip_read_info->rest_read_compressed < uReadThis) uReadThis = (uInt)pfile_in_zip_read_info->rest_read_compressed;
				if (uReadThis == 0) {
					if (reached_eof != 0) *reached_eof = true; return UNZ_EOF;
				}
				if (lufseek(pfile_in_zip_read_info->file, pfile_in_zip_read_info->pos_in_zipfile + pfile_in_zip_read_info->byte_before_the_zipfile, SEEK_SET) != 0) return UNZ_ERRNO;
				if (lufread(pfile_in_zip_read_info->read_buffer, uReadThis, 1, pfile_in_zip_read_info->file) != 1) return UNZ_ERRNO;
				pfile_in_zip_read_info->pos_in_zipfile += uReadThis;
				pfile_in_zip_read_info->rest_read_compressed -= uReadThis;
				pfile_in_zip_read_info->stream.next_in = (Byte*)pfile_in_zip_read_info->read_buffer;
				pfile_in_zip_read_info->stream.avail_in = (uInt)uReadThis;
				//
				if (pfile_in_zip_read_info->encrypted) {
					char *buf = (char*)pfile_in_zip_read_info->stream.next_in;
					for (unsigned int i = 0; i < uReadThis; i++) buf[i] = zdecode(pfile_in_zip_read_info->keys, buf[i]);
				}
			}

			unsigned int uDoEncHead = pfile_in_zip_read_info->encheadleft;
			if (uDoEncHead > pfile_in_zip_read_info->stream.avail_in) uDoEncHead = pfile_in_zip_read_info->stream.avail_in;
			if (uDoEncHead > 0) {
				char bufcrc = pfile_in_zip_read_info->stream.next_in[uDoEncHead - 1];
				pfile_in_zip_read_info->rest_read_uncompressed -= uDoEncHead;
				pfile_in_zip_read_info->stream.avail_in -= uDoEncHead;
				pfile_in_zip_read_info->stream.next_in += uDoEncHead;
				pfile_in_zip_read_info->encheadleft -= uDoEncHead;
				if (pfile_in_zip_read_info->encheadleft == 0) {
					if (bufcrc != pfile_in_zip_read_info->crcenctest) return UNZ_PASSWORD;
				}
			}

			if (pfile_in_zip_read_info->compression_method == 0) {
				uInt uDoCopy, i;
				if (pfile_in_zip_read_info->stream.avail_out < pfile_in_zip_read_info->stream.avail_in) {
					uDoCopy = pfile_in_zip_read_info->stream.avail_out;
				} else {
					uDoCopy = pfile_in_zip_read_info->stream.avail_in;
				}
				for (i = 0; i < uDoCopy; i++) *(pfile_in_zip_read_info->stream.next_out + i) = *(pfile_in_zip_read_info->stream.next_in + i);
				pfile_in_zip_read_info->crc32 = ucrc32(pfile_in_zip_read_info->crc32, pfile_in_zip_read_info->stream.next_out, uDoCopy);
				pfile_in_zip_read_info->rest_read_uncompressed -= uDoCopy;
				pfile_in_zip_read_info->stream.avail_in -= uDoCopy;
				pfile_in_zip_read_info->stream.avail_out -= uDoCopy;
				pfile_in_zip_read_info->stream.next_out += uDoCopy;
				pfile_in_zip_read_info->stream.next_in += uDoCopy;
				pfile_in_zip_read_info->stream.total_out += uDoCopy;
				iRead += uDoCopy;
				if (pfile_in_zip_read_info->rest_read_uncompressed == 0) {
					if (reached_eof != 0) *reached_eof = true;
				}
			} else {
				uLong uTotalOutBefore, uTotalOutAfter;
				const Byte *bufBefore;
				uLong uOutThis;
				int flush = Z_SYNC_FLUSH;
				uTotalOutBefore = pfile_in_zip_read_info->stream.total_out;
				bufBefore = pfile_in_zip_read_info->stream.next_out;
				//
				err = inflate(&pfile_in_zip_read_info->stream, flush);
				//
				uTotalOutAfter = pfile_in_zip_read_info->stream.total_out;
				uOutThis = uTotalOutAfter - uTotalOutBefore;
				pfile_in_zip_read_info->crc32 = ucrc32(pfile_in_zip_read_info->crc32, bufBefore, (uInt)(uOutThis));
				pfile_in_zip_read_info->rest_read_uncompressed -= uOutThis;
				iRead += (uInt)(uTotalOutAfter - uTotalOutBefore);
				if (err == Z_STREAM_END || pfile_in_zip_read_info->rest_read_uncompressed == 0) {
					if (reached_eof != 0) *reached_eof = true;
					return iRead;
				}
				if (err != Z_OK) break;
			}
		}

		if (err == Z_OK) return iRead;
		return err;
	}


	//  Give the current position in uncompressed data
	inline z_off_t unztell(unzFile file) {
		unz_s* s;
		file_in_zip_read_info_s* pfile_in_zip_read_info;
		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		pfile_in_zip_read_info = s->pfile_in_zip_read;

		if (pfile_in_zip_read_info == NULL)
			return UNZ_PARAMERROR;

		return (z_off_t)pfile_in_zip_read_info->stream.total_out;
	}


	//  return 1 if the end of file was reached, 0 elsewhere
	inline int unzeof(unzFile file) {
		unz_s* s;
		file_in_zip_read_info_s* pfile_in_zip_read_info;
		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		pfile_in_zip_read_info = s->pfile_in_zip_read;

		if (pfile_in_zip_read_info == NULL)
			return UNZ_PARAMERROR;

		if (pfile_in_zip_read_info->rest_read_uncompressed == 0)
			return 1;
		else
			return 0;
	}



	//  Read extra field from the current file (opened by unzOpenCurrentFile)
	//  This is the local-header version of the extra field (sometimes, there is
	//    more info in the local-header version than in the central-header)
	//  if buf==NULL, it return the size of the local extra field that can be read
	//  if buf!=NULL, len is the size of the buffer, the extra header is copied in buf.
	//  the return value is the number of bytes copied in buf, or (if <0) the error code
	inline int unzGetLocalExtrafield(unzFile file, voidp buf, unsigned len) {
		unz_s* s;
		file_in_zip_read_info_s* pfile_in_zip_read_info;
		uInt read_now;
		uLong size_to_read;

		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		pfile_in_zip_read_info = s->pfile_in_zip_read;

		if (pfile_in_zip_read_info == NULL)
			return UNZ_PARAMERROR;

		size_to_read = (pfile_in_zip_read_info->size_local_extrafield -
						pfile_in_zip_read_info->pos_local_extrafield);

		if (buf == NULL)
			return (int)size_to_read;

		if (len > size_to_read)
			read_now = (uInt)size_to_read;
		else
			read_now = (uInt)len;

		if (read_now == 0)
			return 0;

		if (lufseek(pfile_in_zip_read_info->file, pfile_in_zip_read_info->offset_local_extrafield + pfile_in_zip_read_info->pos_local_extrafield, SEEK_SET) != 0)
			return UNZ_ERRNO;

		if (lufread(buf, (uInt)size_to_read, 1, pfile_in_zip_read_info->file) != 1)
			return UNZ_ERRNO;

		return (int)read_now;
	}

	//  Close the file in zip opened with unzipOpenCurrentFile
	//  Return UNZ_CRCERROR if all the file was read but the CRC is not good
	inline int unzCloseCurrentFile(unzFile file) {
		int err = UNZ_OK;

		unz_s* s;
		file_in_zip_read_info_s* pfile_in_zip_read_info;
		if (file == NULL)
			return UNZ_PARAMERROR;
		s = (unz_s*)file;
		pfile_in_zip_read_info = s->pfile_in_zip_read;

		if (pfile_in_zip_read_info == NULL)
			return UNZ_PARAMERROR;


		if (pfile_in_zip_read_info->rest_read_uncompressed == 0) {
			if (pfile_in_zip_read_info->crc32 != pfile_in_zip_read_info->crc32_wait)
				err = UNZ_CRCERROR;
		}


		if (pfile_in_zip_read_info->read_buffer != 0) {
			void *buf = pfile_in_zip_read_info->read_buffer;
			zfree(buf);
			pfile_in_zip_read_info->read_buffer = 0;
		}
		pfile_in_zip_read_info->read_buffer = NULL;
		if (pfile_in_zip_read_info->stream_initialised)
			inflateEnd(&pfile_in_zip_read_info->stream);

		pfile_in_zip_read_info->stream_initialised = 0;
		if (pfile_in_zip_read_info != 0) zfree(pfile_in_zip_read_info); // unused pfile_in_zip_read_info=0;

		s->pfile_in_zip_read = NULL;

		return err;
	}


	//  Get the global comment string of the ZipFile, in the szComment buffer.
	//  uSizeBuf is the size of the szComment buffer.
	//  return the number of byte copied or an error code <0
	inline int unzGetGlobalComment(unzFile file, char *szComment, uLong uSizeBuf) { //int err=UNZ_OK;
		unz_s* s;
		uLong uReadThis;
		if (file == NULL) return UNZ_PARAMERROR;
		s = (unz_s*)file;
		uReadThis = uSizeBuf;
		if (uReadThis > s->gi.size_comment) uReadThis = s->gi.size_comment;
		if (lufseek(s->file, s->central_pos + 22, SEEK_SET) != 0) return UNZ_ERRNO;
		if (uReadThis > 0) {
			*szComment = '\0';
			if (lufread(szComment, (uInt)uReadThis, 1, s->file) != 1) return UNZ_ERRNO;
		}
		if ((szComment != NULL) && (uSizeBuf > s->gi.size_comment)) *(szComment + s->gi.size_comment) = '\0';
		return (int)uReadThis;
	}





	inline int unzOpenCurrentFile(unzFile file, const char *password);
	inline int unzReadCurrentFile(unzFile file, void *buf, unsigned len);
	inline int unzCloseCurrentFile(unzFile file);




	class TUnzip {
	public:
		TUnzip(const char *pwd) : uf(0), unzbuf(0), currentfile(-1), czei(-1), password(0) {
			if (pwd != 0) {
				password = new char[strlen(pwd) + 1]; 
#ifdef _MSC_VER
				strcpy_s(password, strlen(pwd) + 1, pwd);
#elif __GNUC__
				strcpy(password, pwd);
#endif
			}
		}
		~TUnzip() {
			if (password != 0) delete[] password; password = 0; if (unzbuf != 0) delete[] unzbuf; unzbuf = 0;
		}

		unzFile uf; int currentfile; ZIPENTRY cze; int czei;
		char *password;
		char *unzbuf;            // lazily created and destroyed, used by Unzip
		TCHAR rootdir[MAX_PATH]; // includes a trailing slash

		inline ZRESULT Open(void *z, unsigned int len, DWORD flags);
		inline ZRESULT Get(int index, ZIPENTRY *ze);
		inline ZRESULT Find(const TCHAR *name, bool ic, int *index, ZIPENTRY *ze);
		inline ZRESULT Unzip(int index, void *dst, unsigned int len, DWORD flags);
		inline ZRESULT SetUnzipBaseDir(const TCHAR *dir);
		inline ZRESULT Close();
	};


	inline ZRESULT TUnzip::Open(void *z, unsigned int len, DWORD flags) {
		if (uf != 0 || currentfile != -1) return ZR_NOTINITED;
		//

#ifdef GetCurrentDirectory
		GetCurrentDirectory(MAX_PATH - 1, rootdir);
#else
		rootdir[0] = '/'; rootdir[1] = 0;
#endif
		TCHAR *lastchar = &rootdir[_tcslen(rootdir) - 1];
		if (*lastchar != '\\' && *lastchar != '/') {
			lastchar[1] = '/'; lastchar[2] = 0;
		}
		//
		if (flags == ZIP_HANDLE) { // test if we can seek on it. We can't use GetFileType(h)==FILE_TYPE_DISK since it's not on CE.
			DWORD res = GetFilePosU((HANDLE)z);
			bool canseek = (res != 0xFFFFFFFF);
			if (!canseek) return ZR_SEEK;
		}
		ZRESULT e; LUFILE *f = lufopen(z, len, flags, &e);
		if (f == NULL) return e;
		uf = unzOpenInternal(f);
		if (uf == 0) return ZR_NOFILE;
		return ZR_OK;
	}

	inline ZRESULT TUnzip::SetUnzipBaseDir(const TCHAR *dir) {

		_tcsncpy_s(rootdir, 260, dir, MAX_PATH - 1);

		TCHAR *lastchar = &rootdir[_tcslen(rootdir) - 1];
		if (*lastchar != '\\' && *lastchar != '/') {
			lastchar[1] = '/'; lastchar[2] = 0;
		}
		return ZR_OK;
	}

	inline ZRESULT TUnzip::Get(int index, ZIPENTRY *ze) {
		if (index < -1 || index >= (int)uf->gi.number_entry) return ZR_ARGS;
		if (currentfile != -1) unzCloseCurrentFile(uf); currentfile = -1;
		if (index == czei && index != -1) {
			memcpy(ze, &cze, sizeof(ZIPENTRY)); return ZR_OK;
		}
		if (index == -1) {
			ze->index = uf->gi.number_entry;
			ze->name[0] = 0;
			ze->attr = 0;

			ze->atime.dwLowDateTime = 0; ze->atime.dwHighDateTime = 0;
			ze->ctime.dwLowDateTime = 0; ze->ctime.dwHighDateTime = 0;
			ze->mtime.dwLowDateTime = 0; ze->mtime.dwHighDateTime = 0;

			ze->comp_size = 0;
			ze->unc_size = 0;
			return ZR_OK;
		}
		if (index < (int)uf->num_file) unzGoToFirstFile(uf);
		while ((int)uf->num_file < index) unzGoToNextFile(uf);
		unz_file_info ufi; char fn[MAX_PATH];
		unzGetCurrentFileInfo(uf, &ufi, fn, MAX_PATH, NULL, 0, NULL, 0);
		// now get the extra header. We do this ourselves, instead of
		// calling unzOpenCurrentFile &c., to avoid allocating more than necessary.
		unsigned int extralen, iSizeVar; unsigned long offset;
		int res = unzlocal_CheckCurrentFileCoherencyHeader(uf, &iSizeVar, &offset, &extralen);
		if (res != UNZ_OK) return ZR_CORRUPT;
		if (lufseek(uf->file, offset, SEEK_SET) != 0) return ZR_READ;
		unsigned char *extra = new unsigned char[extralen];
		if (lufread(extra, 1, (uInt)extralen, uf->file) != extralen) {
			delete[] extra; return ZR_READ;
		}
		//
		ze->index = uf->num_file;
		TCHAR tfn[MAX_PATH];
#ifdef UNICODE
		MultiByteToWideChar(CP_UTF8, 0, fn, -1, tfn, MAX_PATH);
#else
		strcpy(tfn, fn);
#endif
		// As a safety feature: if the zip filename had sneaky stuff
		// like "c:\windows\file.txt" or "\windows\file.txt" or "fred\..\..\..\windows\file.txt"
		// then we get rid of them all. That way, when the programmer does UnzipItem(hz,i,ze.name),
		// it won't be a problem. (If the programmer really did want to get the full evil information,
		// then they can edit out this security feature from here).
		// In particular, we chop off any prefixes that are "c:\" or "\" or "/" or "[stuff]\.." or "[stuff]/.."
		const TCHAR *sfn = tfn;
		for (;;) {
			if (sfn[0] != 0 && sfn[1] == ':') {
				sfn += 2; continue;
			}
			if (sfn[0] == '\\') {
				sfn++; continue;
			}
			if (sfn[0] == '/') {
				sfn++; continue;
			}
			const TCHAR *c;
			c = _tcsstr(sfn, _T("\\..\\")); if (c != 0) {
				sfn = c + 4; continue;
			}
			c = _tcsstr(sfn, _T("\\../")); if (c != 0) {
				sfn = c + 4; continue;
			}
			c = _tcsstr(sfn, _T("/../")); if (c != 0) {
				sfn = c + 4; continue;
			}
			c = _tcsstr(sfn, _T("/..\\")); if (c != 0) {
				sfn = c + 4; continue;
			}
			break;
		}
		_tcsncpy_s(ze->name, 260, sfn, MAX_PATH);


		unsigned long a = ufi.external_fa;
		// zip has an 'attribute' 32bit value. Its lower half is windows stuff
		// its upper half is standard unix stat.st_mode. We'll start trying
		// to read it in unix mode
		bool isdir = (a & 0x40000000) != 0;
		bool readonly = (a & 0x00800000) == 0;
		//bool readable=  (a&0x01000000)!=0; // unused
		//bool executable=(a&0x00400000)!=0; // unused
		bool hidden = false, system = false, archive = true;
		// but in normal hostmodes these are overridden by the lower half...
		int host = ufi.version >> 8;
		if (host == 0 || host == 7 || host == 11 || host == 14) {
			readonly = (a & 0x00000001) != 0;
			hidden = (a & 0x00000002) != 0;
			system = (a & 0x00000004) != 0;
			isdir = (a & 0x00000010) != 0;
			archive = (a & 0x00000020) != 0;
		}
		readonly; hidden; system; isdir; archive;
		ze->attr = 0;

		if (isdir) ze->attr |= FILE_ATTRIBUTE_DIRECTORY;
		if (archive) ze->attr |= FILE_ATTRIBUTE_ARCHIVE;
		if (hidden) ze->attr |= FILE_ATTRIBUTE_HIDDEN;
		if (readonly) ze->attr |= FILE_ATTRIBUTE_READONLY;
		if (system) ze->attr |= FILE_ATTRIBUTE_SYSTEM;

		ze->comp_size = ufi.compressed_size;
		ze->unc_size = ufi.uncompressed_size;
		//
		WORD dostime = (WORD)(ufi.dosDate & 0xFFFF);
		WORD dosdate = (WORD)((ufi.dosDate >> 16) & 0xFFFF);
		FILETIME ftd = dosdatetime2filetime(dosdate, dostime);
		FILETIME ft; LocalFileTimeToFileTime(&ftd, &ft);
		ze->atime = ft; ze->ctime = ft; ze->mtime = ft;
		// the zip will always have at least that dostime. But if it also has
		// an extra header, then we'll instead get the info from that.
		unsigned int epos = 0;
		while (epos + 4 < extralen) {
			char etype[3]; etype[0] = extra[epos + 0]; etype[1] = extra[epos + 1]; etype[2] = 0;
			int size = extra[epos + 2];
			if (strcmp(etype, "UT") != 0) {
				epos += 4 + size; continue;
			}
			int flags = extra[epos + 4];
			bool hasmtime = (flags & 1) != 0;
			bool hasatime = (flags & 2) != 0;
			bool hasctime = (flags & 4) != 0;
			epos += 5;
			if (hasmtime) {
				lutime_t mtime = ((extra[epos + 0]) << 0) | ((extra[epos + 1]) << 8) | ((extra[epos + 2]) << 16) | ((extra[epos + 3]) << 24);
				epos += 4;
				ze->mtime = timet2filetime(mtime);
			}
			if (hasatime) {
				lutime_t atime = ((extra[epos + 0]) << 0) | ((extra[epos + 1]) << 8) | ((extra[epos + 2]) << 16) | ((extra[epos + 3]) << 24);
				epos += 4;
				ze->atime = timet2filetime(atime);
			}
			if (hasctime) {
				lutime_t ctime = ((extra[epos + 0]) << 0) | ((extra[epos + 1]) << 8) | ((extra[epos + 2]) << 16) | ((extra[epos + 3]) << 24);
				epos += 4;
				ze->ctime = timet2filetime(ctime);
			}
			break;
		}
		//
		if (extra != 0) delete[] extra;
		memcpy(&cze, ze, sizeof(ZIPENTRY)); czei = index;
		return ZR_OK;
	}

	inline ZRESULT TUnzip::Find(const TCHAR *tname, bool ic, int *index, ZIPENTRY *ze) {
		char name[MAX_PATH];
#ifdef UNICODE
		WideCharToMultiByte(CP_UTF8, 0, tname, -1, name, MAX_PATH, 0, 0);
#else
		strcpy(name, tname);
#endif
		int res = unzLocateFile(uf, name, ic ? CASE_INSENSITIVE : CASE_SENSITIVE);
		if (res != UNZ_OK) {
			if (index != 0) *index = -1;
			if (ze != NULL) {
				memset(ze, 0, sizeof(ZIPENTRY)); ze->index = -1;
			}
			return ZR_NOTFOUND;
		}
		if (currentfile != -1) unzCloseCurrentFile(uf); currentfile = -1;
		int i = (int)uf->num_file;
		if (index != NULL) *index = i;
		if (ze != NULL) {
			ZRESULT zres = Get(i, ze);
			if (zres != ZR_OK) return zres;
		}
		return ZR_OK;
	}

	inline void EnsureDirectory(const TCHAR *rootdir, const TCHAR *dir) { // first check that rootdir exists. nb. rootdir has a trailing slash
		if (rootdir != 0) {
			TCHAR rd[MAX_PATH]; _tcsncpy_s(rd, 260, rootdir, MAX_PATH); size_t len = _tcslen(rd);
			if (len > 0 && (rd[len - 1] == '/' || rd[len - 1] == '\\')) rd[len - 1] = 0;
			if (!FileExists(rd)) CreateDirectory(rd, 0);
		}
		if (*dir == 0) return;
		const TCHAR *lastslash = dir, *c = lastslash;
		while (*c != 0) {
			if (*c == '/' || *c == '\\') lastslash = c; c++;
		}
		const TCHAR *name = lastslash;
		if (lastslash != dir) {
			TCHAR tmp[MAX_PATH]; memcpy(tmp, dir, sizeof(TCHAR)*(lastslash - dir));
			tmp[lastslash - dir] = 0;
			EnsureDirectory(rootdir, tmp);
			name++;
		}
		TCHAR cd[MAX_PATH]; *cd = 0; if (rootdir != 0) _tcsncpy_s(cd, 260, rootdir, MAX_PATH); cd[MAX_PATH - 1] = 0;
		size_t len = _tcslen(cd); _tcsncpy_s(cd + len, 260 - len, dir, MAX_PATH - len); cd[MAX_PATH - 1] = 0;

		if (!FileExists(cd)) {
			CreateDirectory(cd, 0);
		}
	}



	inline ZRESULT TUnzip::Unzip(int index, void *dst, unsigned int len, DWORD flags) {
		if (flags != ZIP_MEMORY && flags != ZIP_FILENAME && flags != ZIP_HANDLE) return ZR_ARGS;
		if (flags == ZIP_MEMORY) {
			if (index != currentfile) {
				if (currentfile != -1) unzCloseCurrentFile(uf); currentfile = -1;
				if (index >= (int)uf->gi.number_entry) return ZR_ARGS;
				if (index < (int)uf->num_file) unzGoToFirstFile(uf);
				while ((int)uf->num_file < index) unzGoToNextFile(uf);
				unzOpenCurrentFile(uf, password); currentfile = index;
			}
			bool reached_eof;
			int res = unzReadCurrentFile(uf, dst, len, &reached_eof);
			if (res <= 0) {
				unzCloseCurrentFile(uf); currentfile = -1;
			}
			if (reached_eof) return ZR_OK;
			if (res > 0) return ZR_MORE;
			if (res == UNZ_PASSWORD) return ZR_PASSWORD;
			return ZR_FLATE;
		}
		// otherwise we're writing to a handle or a file
		if (currentfile != -1) unzCloseCurrentFile(uf); currentfile = -1;
		if (index >= (int)uf->gi.number_entry) return ZR_ARGS;
		if (index < (int)uf->num_file) unzGoToFirstFile(uf);
		while ((int)uf->num_file < index) unzGoToNextFile(uf);
		ZIPENTRY ze; Get(index, &ze);
		// zipentry=directory is handled specially
		bool isdir = (ze.attr&FILE_ATTRIBUTE_DIRECTORY) != 0;
		if (isdir) {
			if (flags == ZIP_HANDLE) return ZR_OK; // don't do anything
			const TCHAR *dir = (const TCHAR*)dst;
			bool isabsolute = (dir[0] == '/' || dir[0] == '\\' || (dir[0] != 0 && dir[1] == ':'));
			if (isabsolute) EnsureDirectory(0, dir); else EnsureDirectory(rootdir, dir);
			return ZR_OK;
		}
		// otherwise, we write the zipentry to a file/handle
		HANDLE h; TCHAR fn[MAX_PATH]; fn[0] = 0;
		if (flags == ZIP_HANDLE) h = (HANDLE)dst;
		else {
			const TCHAR *ufn = (const TCHAR*)dst;
			// We'll qualify all relative names to our root dir, and leave absolute names as they are
			// ufn="zipfile.txt"  dir=""  name="zipfile.txt"  fn="c:\\currentdir\\zipfile.txt"
			// ufn="dir1/dir2/subfile.txt"  dir="dir1/dir2/"  name="subfile.txt"  fn="c:\\currentdir\\dir1/dir2/subfiles.txt"
			// ufn="\z\file.txt"  dir="\z\"  name="file.txt"  fn="\z\file.txt"
			// This might be a security risk, in the case where we just use the zipentry's name as "ufn", where
			// a malicious zip could unzip itself into c:\windows. Our solution is that GetZipItem (which
			// is how the user retrieve's the file's name within the zip) never returns absolute paths.
			const TCHAR *name = ufn; const TCHAR *c = name; while (*c != 0) {
				if (*c == '/' || *c == '\\') name = c + 1; c++;
			}
			TCHAR dir[MAX_PATH]; _tcsncpy_s(dir, 260, ufn, MAX_PATH); if (name == ufn) *dir = 0; else dir[name - ufn] = 0;
			bool isabsolute = (dir[0] == '/' || dir[0] == '\\' || (dir[0] != 0 && dir[1] == ':'));
			if (isabsolute) {
#ifdef _UNICODE
				std::wostringstream oss;
#else
				std::ostringstream oss;
#endif
				oss << dir << name;
				_tcscpy_s(fn, 260, oss.str().c_str());
				//_tsprintf(fn,_T("%s%s"),dir,name);
				EnsureDirectory(0, dir);
			} else {
#ifdef _UNICODE
				std::wostringstream oss;
#else
				std::ostringstream oss;
#endif
				oss << rootdir << dir << name;
				_tcscpy_s(fn, 260, oss.str().c_str());
				//_tsprintf(fn,_T("%s%s%s"),rootdir,dir,name);
				EnsureDirectory(rootdir, dir);
			}
			//
			h = CreateFile(fn, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, ze.attr, NULL);
		}
		if (h == INVALID_HANDLE_VALUE) return ZR_NOFILE;
		unzOpenCurrentFile(uf, password);
		if (unzbuf == 0) unzbuf = new char[16384]; DWORD haderr = 0;
		//  

		for (; haderr == 0;) {
			bool reached_eof;
			int res = unzReadCurrentFile(uf, unzbuf, 16384, &reached_eof);
			if (res == UNZ_PASSWORD) {
				haderr = ZR_PASSWORD; break;
			}
			if (res < 0) {
				haderr = ZR_FLATE; break;
			}
			if (res > 0) {
				DWORD writ; BOOL bres = WriteFile(h, unzbuf, res, &writ, NULL); if (!bres) {
					haderr = ZR_WRITE; break;
				}
			}

			if (reached_eof) break;
			if (res == 0) {
				haderr = ZR_FLATE; break;
			}
		}
		unzCloseCurrentFile(uf);
		if (!haderr) SetFileTime(h, &ze.ctime, &ze.atime, &ze.mtime); // may fail if it was a pipe
		if (flags != ZIP_HANDLE) CloseHandle(h);
		if (haderr != 0) return haderr;
		return ZR_OK;
	}

	inline ZRESULT TUnzip::Close() {
		if (currentfile != -1) unzCloseCurrentFile(uf); currentfile = -1;
		if (uf != 0) unzClose(uf); uf = 0;
		return ZR_OK;
	}





	__declspec(selectany) ZRESULT lasterrorU = ZR_OK;

	inline unsigned int FormatZipMessageU(ZRESULT code, TCHAR *buf, unsigned int len) {
		if (code == ZR_RECENT) code = lasterrorU;
		const TCHAR *msg = _T("unknown zip result code");
		switch (code) {
			case ZR_OK: msg = _T("Success"); break;
			case ZR_NODUPH: msg = _T("Culdn't duplicate handle"); break;
			case ZR_NOFILE: msg = _T("Couldn't create/open file"); break;
			case ZR_NOALLOC: msg = _T("Failed to allocate memory"); break;
			case ZR_WRITE: msg = _T("Error writing to file"); break;
			case ZR_NOTFOUND: msg = _T("File not found in the zipfile"); break;
			case ZR_MORE: msg = _T("Still more data to unzip"); break;
			case ZR_CORRUPT: msg = _T("Zipfile is corrupt or not a zipfile"); break;
			case ZR_READ: msg = _T("Error reading file"); break;
			case ZR_PASSWORD: msg = _T("Correct password required"); break;
			case ZR_ARGS: msg = _T("Caller: faulty arguments"); break;
			case ZR_PARTIALUNZ: msg = _T("Caller: the file had already been partially unzipped"); break;
			case ZR_NOTMMAP: msg = _T("Caller: can only get memory of a memory zipfile"); break;
			case ZR_MEMSIZE: msg = _T("Caller: not enough space allocated for memory zipfile"); break;
			case ZR_FAILED: msg = _T("Caller: there was a previous error"); break;
			case ZR_ENDED: msg = _T("Caller: additions to the zip have already been ended"); break;
			case ZR_ZMODE: msg = _T("Caller: mixing creation and opening of zip"); break;
			case ZR_NOTINITED: msg = _T("Zip-bug: internal initialisation not completed"); break;
			case ZR_SEEK: msg = _T("Zip-bug: trying to seek the unseekable"); break;
			case ZR_MISSIZE: msg = _T("Zip-bug: the anticipated size turned out wrong"); break;
			case ZR_NOCHANGE: msg = _T("Zip-bug: tried to change mind, but not allowed"); break;
			case ZR_FLATE: msg = _T("Zip-bug: an internal error during flation"); break;
		}
		unsigned int mlen = (unsigned int)_tcslen(msg);
		if (buf == 0 || len == 0) return mlen;
		unsigned int n = mlen; if (n + 1 > len) n = len - 1;
		//_tcsncpy_s(buf,msg,n); 
		memcpy(buf, msg, n);
		buf[n] = 0;
		return mlen;
	}


	typedef struct {
		DWORD flag;
		TUnzip *unz;
	} TUnzipHandleData;

	inline HZIP OpenZipInternal(void *z, unsigned int len, DWORD flags, const char *password) {
		TUnzip *unz = new TUnzip(password);
		lasterrorU = unz->Open(z, len, flags);
		if (lasterrorU != ZR_OK) {
			delete unz; return 0;
		}
		TUnzipHandleData *han = new TUnzipHandleData;
		han->flag = 1; han->unz = unz; return (HZIP)han;
	}
	inline HZIP OpenZipHandle(HANDLE h, const char *password) {
		return OpenZipInternal((void*)h, 0, ZIP_HANDLE, password);
	}
	inline HZIP OpenZip(const TCHAR *fn, const char *password) {
		return OpenZipInternal((void*)fn, 0, ZIP_FILENAME, password);
	}
	inline HZIP OpenZip(void *z, unsigned int len, const char *password) {
		return OpenZipInternal(z, len, ZIP_MEMORY, password);
	}


	inline ZRESULT GetZipItem(HZIP hz, int index, ZIPENTRY *ze) {
		ze->index = 0; *ze->name = 0; ze->unc_size = 0;
		if (hz == 0) {
			lasterrorU = ZR_ARGS; return ZR_ARGS;
		}
		TUnzipHandleData *han = (TUnzipHandleData*)hz;
		if (han->flag != 1) {
			lasterrorU = ZR_ZMODE; return ZR_ZMODE;
		}
		TUnzip *unz = han->unz;
		lasterrorU = unz->Get(index, ze);
		return lasterrorU;
	}

	inline ZRESULT FindZipItem(HZIP hz, const TCHAR *name, bool ic, int *index, ZIPENTRY *ze) {
		if (hz == 0) {
			lasterrorU = ZR_ARGS; return ZR_ARGS;
		}
		TUnzipHandleData *han = (TUnzipHandleData*)hz;
		if (han->flag != 1) {
			lasterrorU = ZR_ZMODE; return ZR_ZMODE;
		}
		TUnzip *unz = han->unz;
		lasterrorU = unz->Find(name, ic, index, ze);
		return lasterrorU;
	}

	inline ZRESULT UnzipItemInternal(HZIP hz, int index, void *dst, unsigned int len, DWORD flags) {
		if (hz == 0) {
			lasterrorU = ZR_ARGS; return ZR_ARGS;
		}
		TUnzipHandleData *han = (TUnzipHandleData*)hz;
		if (han->flag != 1) {
			lasterrorU = ZR_ZMODE; return ZR_ZMODE;
		}
		TUnzip *unz = han->unz;
		lasterrorU = unz->Unzip(index, dst, len, flags);
		return lasterrorU;
	}
	inline ZRESULT UnzipItemHandle(HZIP hz, int index, HANDLE h) {
		return UnzipItemInternal(hz, index, (void*)h, 0, ZIP_HANDLE);
	}
	inline ZRESULT UnzipItem(HZIP hz, int index, const TCHAR *fn) {
		return UnzipItemInternal(hz, index, (void*)fn, 0, ZIP_FILENAME);
	}
	inline ZRESULT UnzipItem(HZIP hz, int index, void *z, unsigned int len) {
		return UnzipItemInternal(hz, index, z, len, ZIP_MEMORY);
	}

	inline ZRESULT SetUnzipBaseDir(HZIP hz, const TCHAR *dir) {
		if (hz == 0) {
			lasterrorU = ZR_ARGS; return ZR_ARGS;
		}
		TUnzipHandleData *han = (TUnzipHandleData*)hz;
		if (han->flag != 1) {
			lasterrorU = ZR_ZMODE; return ZR_ZMODE;
		}
		TUnzip *unz = han->unz;
		lasterrorU = unz->SetUnzipBaseDir(dir);
		return lasterrorU;
	}


	inline ZRESULT CloseZipU(HZIP hz) {
		if (hz == 0) {
			lasterrorU = ZR_ARGS; return ZR_ARGS;
		}
		TUnzipHandleData *han = (TUnzipHandleData*)hz;
		if (han->flag != 1) {
			lasterrorU = ZR_ZMODE; return ZR_ZMODE;
		}
		TUnzip *unz = han->unz;
		lasterrorU = unz->Close();
		delete unz;
		delete han;
		return lasterrorU;
	}

	inline bool IsZipHandleU(HZIP hz) {
		if (hz == 0) return false;
		TUnzipHandleData *han = (TUnzipHandleData*)hz;
		return (han->flag == 1);
	}
}
//Zip part
namespace ispring_3rdparty {
#ifndef _zip_H
#define _zip_H
	// THIS FILE is almost entirely based upon code by info-zip.
	// It has been modified by Lucian Wischik. The modifications
	// were a complete rewrite of the bit of code that generates the
	// layout of the zipfile, and support for zipping to/from memory
	// or handles or pipes or pagefile or diskfiles, encryption, unicode.
	// The original code may be found at http://www.info-zip.org
	// The original copyright text follows.
	//
	//
	//
	// This is version 1999-Oct-05 of the Info-ZIP copyright and license.
	// The definitive version of this document should be available at
	// ftp://ftp.cdrom.com/pub/infozip/license.html indefinitely.
	//
	// Copyright (c) 1990-1999 Info-ZIP.  All rights reserved.
	//
	// For the purposes of this copyright and license, "Info-ZIP" is defined as
	// the following set of individuals:
	//
	//   Mark Adler, John Bush, Karl Davis, Harald Denker, Jean-Michel Dubois,
	//   Jean-loup Gailly, Hunter Goatley, Ian Gorman, Chris Herborth, Dirk Haase,
	//   Greg Hartwig, Robert Heath, Jonathan Hudson, Paul Kienitz, David Kirschbaum,
	//   Johnny Lee, Onno van der Linden, Igor Mandrichenko, Steve P. Miller,
	//   Sergio Monesi, Keith Owens, George Petrov, Greg Roelofs, Kai Uwe Rommel,
	//   Steve Salisbury, Dave Smith, Christian Spieler, Antoine Verheijen,
	//   Paul von Behren, Rich Wales, Mike White
	//
	// This software is provided "as is," without warranty of any kind, express
	// or implied.  In no event shall Info-ZIP or its contributors be held liable
	// for any direct, indirect, incidental, special or consequential damages
	// arising out of the use of or inability to use this software.
	//
	// Permission is granted to anyone to use this software for any purpose,
	// including commercial applications, and to alter it and redistribute it
	// freely, subject to the following restrictions:
	//
	//    1. Redistributions of source code must retain the above copyright notice,
	//       definition, disclaimer, and this list of conditions.
	//
	//    2. Redistributions in binary form must reproduce the above copyright
	//       notice, definition, disclaimer, and this list of conditions in
	//       documentation and/or other materials provided with the distribution.
	//
	//    3. Altered versions--including, but not limited to, ports to new operating
	//       systems, existing ports with new graphical interfaces, and dynamic,
	//       shared, or static library versions--must be plainly marked as such
	//       and must not be misrepresented as being the original source.  Such
	//       altered versions also must not be misrepresented as being Info-ZIP
	//       releases--including, but not limited to, labeling of the altered
	//       versions with the names "Info-ZIP" (or any variation thereof, including,
	//       but not limited to, different capitalizations), "Pocket UnZip," "WiZ"
	//       or "MacZip" without the explicit permission of Info-ZIP.  Such altered
	//       versions are further prohibited from misrepresentative use of the
	//       Zip-Bugs or Info-ZIP e-mail addresses or of the Info-ZIP URL(s).
	//
	//    4. Info-ZIP retains the right to use the names "Info-ZIP," "Zip," "UnZip,"
	//       "WiZ," "Pocket UnZip," "Pocket Zip," and "MacZip" for its own source and
	//       binary releases.
	//
#ifndef _unzip_H
	DECLARE_HANDLE(HZIP);
#endif

	typedef DWORD ZRESULT;



	inline HZIP CreateZip(const TCHAR *fn, const char *password);
	inline HZIP CreateZip(void *buf, unsigned int len, const char *password);
	inline HZIP CreateZipHandle(HANDLE h, const char *password);


	inline ZRESULT ZipAdd(HZIP hz, const TCHAR *dstzn, const TCHAR *fn);
	inline ZRESULT ZipAdd(HZIP hz, const TCHAR *dstzn, void *src, unsigned int len);
	inline ZRESULT ZipAddHandle(HZIP hz, const TCHAR *dstzn, HANDLE h);
	inline ZRESULT ZipAddHandle(HZIP hz, const TCHAR *dstzn, HANDLE h, unsigned int len);
	inline ZRESULT ZipAddFolder(HZIP hz, const TCHAR *dstzn);
	inline ZRESULT ZipGetMemory(HZIP hz, void **buf, unsigned long *len);
	inline ZRESULT CloseZip(HZIP hz);
	inline unsigned int FormatZipMessage(ZRESULT code, TCHAR *buf, unsigned int len);

	// These are the result codes:
#define ZR_OK         0x00000000     // nb. the pseudo-code zr-recent is never returned,
#define ZR_RECENT     0x00000001     // but can be passed to FormatZipMessage.
	// The following come from general system stuff (e.g. files not openable)
#define ZR_GENMASK    0x0000FF00
#define ZR_NODUPH     0x00000100     // couldn't duplicate the handle
#define ZR_NOFILE     0x00000200     // couldn't create/open the file
#define ZR_NOALLOC    0x00000300     // failed to allocate some resource
#define ZR_WRITE      0x00000400     // a general error writing to the file
#define ZR_NOTFOUND   0x00000500     // couldn't find that file in the zip
#define ZR_MORE       0x00000600     // there's still more data to be unzipped
#define ZR_CORRUPT    0x00000700     // the zipfile is corrupt or not a zipfile
#define ZR_READ       0x00000800     // a general error reading the file
	// The following come from mistakes on the part of the caller
#define ZR_CALLERMASK 0x00FF0000
#define ZR_ARGS       0x00010000     // general mistake with the arguments
#define ZR_NOTMMAP    0x00020000     // tried to ZipGetMemory, but that only works on mmap zipfiles, which yours wasn't
#define ZR_MEMSIZE    0x00030000     // the memory size is too small
#define ZR_FAILED     0x00040000     // the thing was already failed when you called this function
#define ZR_ENDED      0x00050000     // the zip creation has already been closed
#define ZR_MISSIZE    0x00060000     // the indicated input file size turned out mistaken
#define ZR_PARTIALUNZ 0x00070000     // the file had already been partially unzipped
#define ZR_ZMODE      0x00080000     // tried to mix creating/opening a zip 
	// The following come from bugs within the zip library itself
#define ZR_BUGMASK    0xFF000000
#define ZR_NOTINITED  0x01000000     // initialisation didn't work
#define ZR_SEEK       0x02000000     // trying to seek in an unseekable file
#define ZR_NOCHANGE   0x04000000     // changed its mind on storage, but not allowed
#define ZR_FLATE      0x05000000     // an internal error in the de/inflation code

	inline ZRESULT CloseZipZ(HZIP hz);
	inline unsigned int FormatZipMessageZ(ZRESULT code, char *buf, unsigned int len);
	inline bool IsZipHandleZ(HZIP hz);
#ifdef _unzip_H
#undef CloseZip
#define CloseZip(hz) (IsZipHandleZ(hz)?CloseZipZ(hz):CloseZipU(hz))
#else
#define CloseZip CloseZipZ
#define FormatZipMessage FormatZipMessageZ
#endif

#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tchar.h>
#include <ctype.h>
#include <stdio.h>

	typedef unsigned char uch;      // unsigned 8-bit value
	typedef unsigned short ush;     // unsigned 16-bit value
	typedef unsigned long ulg;      // unsigned 32-bit value
	typedef size_t extent;          // file size
	typedef unsigned Pos;   // must be at least 32 bits
	typedef unsigned IPos; // A Pos is an index in the character window. Pos is used only for parameter passing

#ifndef EOF
#define EOF (-1)
#endif

						   // Error return values.  The values 0..4 and 12..18 follow the conventions
						   // of PKZIP.   The values 4..10 are all assigned to "insufficient memory"
						   // by PKZIP, so the codes 5..10 are used here for other purposes.
#define ZE_MISS         -1      // used by procname(), zipbare()
#define ZE_OK           0       // success
#define ZE_EOF          2       // unexpected end of zip file
#define ZE_FORM         3       // zip file structure error
#define ZE_MEM          4       // out of memory
#define ZE_LOGIC        5       // internal logic error
#define ZE_BIG          6       // entry too large to split
#define ZE_NOTE         7       // invalid comment format
#define ZE_TEST         8       // zip test (-T) failed or out of memory
#define ZE_ABORT        9       // user interrupt or termination
#define ZE_TEMP         10      // error using a temp file
#define ZE_READ         11      // read or seek error
#define ZE_NONE         12      // nothing to do
#define ZE_NAME         13      // missing or empty zip file
#define ZE_WRITE        14      // error writing to a file
#define ZE_CREAT        15      // couldn't open to write
#define ZE_PARMS        16      // bad command line
#define ZE_OPEN         18      // could not open a specified file to read
#define ZE_MAXERR       18      // the highest error number
						   // internal file attribute
#define UNKNOWN (-1)
#define BINARY  0
#define ASCII   1
#define BEST -1                 // Use best method (deflation or store)
#define STORE 0                 // Store method
#define DEFLATE 8               // Deflation method
#define CRCVAL_INITIAL  0L
						   // MSDOS file or directory attributes
#define MSDOS_HIDDEN_ATTR 0x02
#define MSDOS_DIR_ATTR 0x10
						   // Lengths of headers after signatures in bytes
#define LOCHEAD 26
#define CENHEAD 42
#define ENDHEAD 18
						   // Definitions for extra field handling:
#define EB_HEADSIZE       4     /* length of a extra field block header */
#define EB_LEN            2     /* offset of data length field in header */
#define EB_UT_MINLEN      1     /* minimal UT field contains Flags byte */
#define EB_UT_FLAGS       0     /* byte offset of Flags field */
#define EB_UT_TIME1       1     /* byte offset of 1st time value */
#define EB_UT_FL_MTIME    (1 << 0)      /* mtime present */
#define EB_UT_FL_ATIME    (1 << 1)      /* atime present */
#define EB_UT_FL_CTIME    (1 << 2)      /* ctime present */
#define EB_UT_LEN(n)      (EB_UT_MINLEN + 4 * (n))
#define EB_L_UT_SIZE    (EB_HEADSIZE + EB_UT_LEN(3))
#define EB_C_UT_SIZE    (EB_HEADSIZE + EB_UT_LEN(1))
						   // Macros for writing machine integers to little-endian format
#define PUTSH(a,f) {char _putsh_c=(char)((a)&0xff); wfunc(param,&_putsh_c,1); _putsh_c=(char)((a)>>8); wfunc(param,&_putsh_c,1);}
#define PUTLG(a,f) {PUTSH((a) & 0xffff,(f)) PUTSH((a) >> 16,(f))}
						   // -- Structure of a ZIP file --
						   // Signatures for zip file information headers
#define LOCSIG     0x04034b50L
#define CENSIG     0x02014b50L
#define ENDSIG     0x06054b50L
#define EXTLOCSIG  0x08074b50L

#define MIN_MATCH  3
#define MAX_MATCH  258
						   // The minimum and maximum match lengths

#define WSIZE  (0x8000)
#define MIN_LOOKAHEAD (MAX_MATCH+MIN_MATCH+1)

#define _ZIP_MAX_DIST  (WSIZE-MIN_LOOKAHEAD)

#define ZIP_HANDLE   1
#define ZIP_FILENAME 2
#define ZIP_MEMORY   3
#define ZIP_FOLDER   4

#define MAX_BITS 15
#define MAX_BL_BITS 7
#define LENGTH_CODES 29
#define LITERALS  256
#define END_BLOCK 256
#define L_CODES (LITERALS+1+LENGTH_CODES)
#define D_CODES   30
#define BL_CODES  19
#define STORED_BLOCK 0
#define STATIC_TREES 1
#define DYN_TREES    2
#define LIT_BUFSIZE  0x8000
#define DIST_BUFSIZE  LIT_BUFSIZE
#define REP_3_6      16
#define REPZ_3_10    17
#define REPZ_11_138  18
#define HEAP_SIZE (2*L_CODES+1)
#define Buf_size (8 * 2*sizeof(char))
#define PUTSHORT(state,w) \
{ if (state.bs.out_offset >= state.bs.out_size-1) \
    state.flush_outbuf(state.param,state.bs.out_buf, &state.bs.out_offset); \
  state.bs.out_buf[state.bs.out_offset++] = (char) ((w) & 0xff); \
  state.bs.out_buf[state.bs.out_offset++] = (char) ((ush)(w) >> 8); \
}
#define PUTBYTE(state,b) \
{ if (state.bs.out_offset >= state.bs.out_size) \
    state.flush_outbuf(state.param,state.bs.out_buf, &state.bs.out_offset); \
  state.bs.out_buf[state.bs.out_offset++] = (char) (b); \
}
#define HASH_BITS  15
#define HASH_SIZE (unsigned)(1<<HASH_BITS)
#define HASH_MASK (HASH_SIZE-1)
#define WMASK     (WSIZE-1)
#define Z_NIL 0
#define Z_FAST 4
#define Z_SLOW 2
#define Z_TOO_FAR 4096
#define Z_EQUAL 0
#define H_SHIFT  ((HASH_BITS+MIN_MATCH-1)/MIN_MATCH)
#define max_insert_length  max_lazy_match

	const int extra_lbits[LENGTH_CODES] // extra bits for each length code
		= { 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0 };

	const int extra_dbits[D_CODES] // extra bits for each distance code
		= { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };

	const int extra_blbits[BL_CODES]// extra bits for each bit length code
		= { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7 };

	const uch bl_order[BL_CODES] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };

	typedef struct config {
		ush good_length; // reduce lazy search above this match length
		ush max_lazy;    // do not perform lazy search above this match length
		ush nice_length; // quit search above this match length
		ush max_chain;
	} config;
	const config configuration_table[10] = {
		//  good lazy nice chain
		{ 0,    0,  0,    0 },  // 0 store only
		{ 4,    4,  8,    4 },  // 1 maximum speed, no lazy matches
		{ 4,    5, 16,    8 },  // 2
		{ 4,    6, 32,   32 },  // 3
		{ 4,    4, 16,   16 },  // 4 lazy matches */
		{ 8,   16, 32,   32 },  // 5
		{ 8,   16, 128, 128 },  // 6
		{ 8,   32, 128, 256 },  // 7
		{ 32, 128, 258, 1024 }, // 8
		{ 32, 258, 258, 4096 } };// 9 maximum compression */

	typedef struct ct_data {
		union {
			ush  freq;       // frequency count
			ush  code;       // bit string
		} fc;
		union {
			ush  dad;        // father node in Huffman tree
			ush  len;        // length of bit string
		} dl;
	} ct_data;

	typedef struct tree_desc {
		ct_data *dyn_tree;      // the dynamic tree
		ct_data *static_tree;   // corresponding static tree or NULL
		const int *extra_bits;  // extra bits for each code or NULL
		int     extra_base;     // base index for extra_bits
		int     elems;          // max number of elements in the tree
		int     max_length;     // max bit length for the codes
		int     max_code;       // largest code with non zero frequency
	} tree_desc;

	class TTreeState {
	public:
		inline TTreeState();

		ct_data dyn_ltree[HEAP_SIZE];    // literal and length tree
		ct_data dyn_dtree[2 * D_CODES + 1];  // distance tree
		ct_data static_ltree[L_CODES + 2]; // the static literal tree...
										   // ... Since the bit lengths are imposed, there is no need for the L_CODES
										   // extra codes used during heap construction. However the codes 286 and 287
										   // are needed to build a canonical tree (see ct_init below).
		ct_data static_dtree[D_CODES]; // the static distance tree...
									   // ... (Actually a trivial tree since all codes use 5 bits.)
		ct_data bl_tree[2 * BL_CODES + 1];  // Huffman tree for the bit lengths

		tree_desc l_desc;
		tree_desc d_desc;
		tree_desc bl_desc;

		ush bl_count[MAX_BITS + 1];  // number of codes at each bit length for an optimal tree

		int heap[2 * L_CODES + 1]; // heap used to build the Huffman trees
		int heap_len;               // number of elements in the heap
		int heap_max;               // element of largest frequency
									// The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
									// The same heap array is used to build all trees.

		uch depth[2 * L_CODES + 1];
		// Depth of each subtree used as tie breaker for trees of equal frequency

		uch length_code[MAX_MATCH - MIN_MATCH + 1];
		// length code for each normalized match length (0 == MIN_MATCH)

		uch dist_code[512];
		// distance codes. The first 256 values correspond to the distances
		// 3 .. 258, the last 256 values correspond to the top 8 bits of
		// the 15 bit distances.

		int base_length[LENGTH_CODES];
		// First normalized length for each code (0 = MIN_MATCH)

		int base_dist[D_CODES];
		// First normalized distance for each code (0 = distance of 1)

		uch l_buf[LIT_BUFSIZE];  // buffer for literals/lengths
		ush d_buf[DIST_BUFSIZE]; // buffer for distances

		uch flag_buf[(LIT_BUFSIZE / 8)];
		// flag_buf is a bit array distinguishing literals from lengths in
		// l_buf, and thus indicating the presence or absence of a distance.

		unsigned last_lit;    // running index in l_buf
		unsigned last_dist;   // running index in d_buf
		unsigned last_flags;  // running index in flag_buf
		uch flags;            // current flags not yet saved in flag_buf
		uch flag_bit;         // current bit used in flags
							  // bits are filled in flags starting at bit 0 (least significant).
							  // Note: these flags are overkill in the current code since we don't
							  // take advantage of DIST_BUFSIZE == LIT_BUFSIZE.

		ulg opt_len;          // bit length of current block with optimal trees
		ulg static_len;       // bit length of current block with static trees

		ulg cmpr_bytelen;     // total byte length of compressed file
		ulg cmpr_len_bits;    // number of bits past 'cmpr_bytelen'

		ulg input_len;        // total byte length of input file
							  // input_len is for debugging only since we can get it by other means.

		ush *file_type;       // pointer to UNKNOWN, BINARY or ASCII
							  //  int *file_method;     // pointer to DEFLATE or STORE
	};

	TTreeState::TTreeState() {
		tree_desc a = { dyn_ltree, static_ltree, extra_lbits, LITERALS + 1, L_CODES, MAX_BITS, 0 };  l_desc = a;
		tree_desc b = { dyn_dtree, static_dtree, extra_dbits, 0,          D_CODES, MAX_BITS, 0 };  d_desc = b;
		tree_desc c = { bl_tree, NULL,       extra_blbits, 0,         BL_CODES, MAX_BL_BITS, 0 };  bl_desc = c;
		last_lit = 0;
		last_dist = 0;
		last_flags = 0;
	}



	class TBitState {
	public:

		int flush_flg;
		//
		unsigned bi_buf;
		// Output buffer. bits are inserted starting at the bottom (least significant
		// bits). The width of bi_buf must be at least 16 bits.
		int bi_valid;
		// Number of valid bits in bi_buf.  All bits above the last valid bit
		// are always zero.
		char *out_buf;
		// Current output buffer.
		unsigned out_offset;
		// Current offset in output buffer.
		// On 16 bit machines, the buffer is limited to 64K.
		unsigned out_size;
		// Size of current output buffer
		ulg bits_sent;   // bit length of the compressed data  only needed for debugging???
	};







	class TDeflateState {
	public:
		inline TDeflateState() {
			window_size = 0;
		}

		uch    window[2L * WSIZE];
		// Sliding window. Input bytes are read into the second half of the window,
		// and move to the first half later to keep a dictionary of at least WSIZE
		// bytes. With this organization, matches are limited to a distance of
		// WSIZE-MAX_MATCH bytes, but this ensures that IO is always
		// performed with a length multiple of the block size. Also, it limits
		// the window size to 64K, which is quite useful on MSDOS.
		// To do: limit the window size to WSIZE+CBSZ if SMALL_MEM (the code would
		// be less efficient since the data would have to be copied WSIZE/CBSZ times)
		Pos    prev[WSIZE];
		// Link to older string with same hash index. To limit the size of this
		// array to 64K, this link is maintained only for the last 32K strings.
		// An index in this array is thus a window index modulo 32K.
		Pos    head[HASH_SIZE];
		// Heads of the hash chains or NIL. If your compiler thinks that
		// HASH_SIZE is a dynamic value, recompile with -DDYN_ALLOC.

		ulg window_size;
		// window size, 2*WSIZE except for MMAP or BIG_MEM, where it is the
		// input file length plus MIN_LOOKAHEAD.

		long block_start;
		// window position at the beginning of the current output block. Gets
		// negative when the window is moved backwards.

		int sliding;
		// Set to false when the input file is already in memory

		unsigned ins_h;  // hash index of string to be inserted

		unsigned int prev_length;
		// Length of the best match at previous step. Matches not greater than this
		// are discarded. This is used in the lazy match evaluation.

		unsigned strstart;         // start of string to insert
		unsigned match_start; // start of matching string
		int      eofile;           // flag set at end of input file
		unsigned lookahead;        // number of valid bytes ahead in window

		unsigned max_chain_length;
		// To speed up deflation, hash chains are never searched beyond this length.
		// A higher limit improves compression ratio but degrades the speed.

		unsigned int max_lazy_match;
		// Attempt to find a better match only when the current match is strictly
		// smaller than this value. This mechanism is used only for compression
		// levels >= 4.

		unsigned good_match;
		// Use a faster search when the previous match is longer than this

		int nice_match; // Stop searching when current match exceeds this
	};
#ifndef _unzip_H
	typedef long lutime_t;       // define it ourselves since we don't include time.h
#endif
	typedef struct iztimes {
		lutime_t atime, mtime, ctime;
	} iztimes; // access, modify, create times

	typedef struct zlist {
		ush vem, ver, flg, how;       // See central header in zipfile.c for what vem..off are
		ulg tim, crc, siz, len;
		extent nam, ext, cext, com;   // offset of ext must be >= LOCHEAD
		ush dsk, att, lflg;           // offset of lflg must be >= LOCHEAD
		ulg atx, off;
		char name[MAX_PATH];          // File name in zip file
		char *extra;                  // Extra field (set only if ext != 0)
		char *cextra;                 // Extra in central (set only if cext != 0)
		char *comment;                // Comment (set only if com != 0)
		char iname[MAX_PATH];         // Internal file name after cleanup
		char zname[MAX_PATH];         // External version of internal name
		int mark;                     // Marker for files to operate on
		int trash;                    // Marker for files to delete
		int dosflag;                  // Set to force MSDOS file attributes
		struct zlist *nxt;        // Pointer to next header in list
	} TZipFileInfo;


	struct TState;
	typedef unsigned(*READFUNC)(TState &state, char *buf, unsigned size);
	typedef unsigned(*FLUSHFUNC)(void *param, const char *buf, unsigned *size);
	typedef unsigned(*WRITEFUNC)(void *param, const char *buf, unsigned size);
	struct TState {
		void *param;
		int level; bool seekable;
		READFUNC readfunc; FLUSHFUNC flush_outbuf;
		TTreeState ts; TBitState bs; TDeflateState ds;
		const char *err;
	};
	inline void filetime2dosdatetime(const FILETIME ft, WORD *dosdate, WORD *dostime) { // date: bits 0-4 are day of month 1-31. Bits 5-8 are month 1..12. Bits 9-15 are year-1980
																						// time: bits 0-4 are seconds/2, bits 5-10 are minute 0..59. Bits 11-15 are hour 0..23
		SYSTEMTIME st;
		FileTimeToSystemTime(&ft, &st);
		*dosdate = (WORD)(((st.wYear - 1980) & 0x7f) << 9);
		*dosdate |= (WORD)((st.wMonth & 0xf) << 5);
		*dosdate |= (WORD)((st.wDay & 0x1f));
		*dostime = (WORD)((st.wHour & 0x1f) << 11);
		*dostime |= (WORD)((st.wMinute & 0x3f) << 5);
		*dostime |= (WORD)((st.wSecond * 2) & 0x1f);
	}

	inline lutime_t filetime2timet(const FILETIME ft) {
		LONGLONG i = *(LONGLONG*)&ft;
		return (lutime_t)((i - 116444736000000000LL) / 10000000LL);
	}

	inline void GetNow(lutime_t *pft, WORD *dosdate, WORD *dostime) {
		SYSTEMTIME st; GetLocalTime(&st);
		FILETIME ft;   SystemTimeToFileTime(&st, &ft);
		filetime2dosdatetime(ft, dosdate, dostime);
		*pft = filetime2timet(ft);
	}

	inline DWORD GetFilePosZ(HANDLE hfout) {
		return SetFilePointer(hfout, 0, 0, FILE_CURRENT);
	}


	inline ZRESULT GetFileInfo(HANDLE hf, ulg *attr, long *size, iztimes *times, ulg *timestamp) { // The handle must be a handle to a file
																								   // The date and time is returned in a long with the date most significant to allow
																								   // unsigned integer comparison of absolute times. The attributes have two
																								   // high bytes unix attr, and two low bytes a mapping of that to DOS attr.
																								   //struct stat s; int res=stat(fn,&s); if (res!=0) return false;
																								   // translate windows file attributes into zip ones.
		BY_HANDLE_FILE_INFORMATION bhi; BOOL res = GetFileInformationByHandle(hf, &bhi);
		if (!res) return ZR_NOFILE;
		DWORD fa = bhi.dwFileAttributes; ulg a = 0;
		// Zip uses the lower word for its interpretation of windows stuff
		if (fa&FILE_ATTRIBUTE_READONLY) a |= 0x01;
		if (fa&FILE_ATTRIBUTE_HIDDEN)   a |= 0x02;
		if (fa&FILE_ATTRIBUTE_SYSTEM)   a |= 0x04;
		if (fa&FILE_ATTRIBUTE_DIRECTORY)a |= 0x10;
		if (fa&FILE_ATTRIBUTE_ARCHIVE)  a |= 0x20;
		// It uses the upper word for standard unix attr, which we manually construct
		if (fa&FILE_ATTRIBUTE_DIRECTORY)a |= 0x40000000;  // directory
		else a |= 0x80000000;  // normal file
		a |= 0x01000000;      // readable
		if (fa&FILE_ATTRIBUTE_READONLY) {
		} else a |= 0x00800000; // writeable
								// now just a small heuristic to check if it's an executable:
		DWORD red, hsize = GetFileSize(hf, NULL); if (hsize > 40) {
			SetFilePointer(hf, 0, NULL, FILE_BEGIN); unsigned short magic; ReadFile(hf, &magic, sizeof(magic), &red, NULL);
			SetFilePointer(hf, 36, NULL, FILE_BEGIN); unsigned long hpos;  ReadFile(hf, &hpos, sizeof(hpos), &red, NULL);
			if (magic == 0x54AD && hsize > hpos + 4 + 20 + 28) {
				SetFilePointer(hf, hpos, NULL, FILE_BEGIN); unsigned long signature; ReadFile(hf, &signature, sizeof(signature), &red, NULL);
				if (signature == IMAGE_DOS_SIGNATURE || signature == IMAGE_OS2_SIGNATURE
					|| signature == IMAGE_OS2_SIGNATURE_LE || signature == IMAGE_NT_SIGNATURE) {
					a |= 0x00400000; // executable
				}
			}
		}
		//
		if (attr != NULL) *attr = a;
		if (size != NULL) *size = hsize;
		if (times != NULL) { // lutime_t is 32bit number of seconds elapsed since 0:0:0GMT, Jan1, 1970.
							 // but FILETIME is 64bit number of 100-nanosecs since Jan1, 1601
			times->atime = filetime2timet(bhi.ftLastAccessTime);
			times->mtime = filetime2timet(bhi.ftLastWriteTime);
			times->ctime = filetime2timet(bhi.ftCreationTime);
		}
		if (timestamp != NULL) {
			WORD dosdate, dostime;
			filetime2dosdatetime(bhi.ftLastWriteTime, &dosdate, &dostime);
			*timestamp = (WORD)dostime | (((DWORD)dosdate) << 16);
		}
		return ZR_OK;
	}
	// ----------------------------------------------------------------------





	inline void Assert(TState &state, bool cond, const char *msg) {
		if (cond) return;
		state.err = msg;
	}
	inline void Trace(const char *x, ...) {
		va_list paramList; va_start(paramList, x); paramList; va_end(paramList);
	}
	inline void Tracec(bool, const char *x, ...) {
		va_list paramList; va_start(paramList, x); paramList; va_end(paramList);
	}



	// ===========================================================================
	// Local (static) routines in this file.
	//

	inline void init_block(TState &);
	inline void pqdownheap(TState &, ct_data *tree, int k);
	inline void gen_bitlen(TState &, tree_desc *desc);
	inline void gen_codes(TState &state, ct_data *tree, int max_code);
	inline void build_tree(TState &, tree_desc *desc);
	inline void scan_tree(TState &, ct_data *tree, int max_code);
	inline void send_tree(TState &state, ct_data *tree, int max_code);
	inline int  build_bl_tree(TState &);
	inline void send_all_trees(TState &state, int lcodes, int dcodes, int blcodes);
	inline void compress_block(TState &state, ct_data *ltree, ct_data *dtree);
	inline void set_file_type(TState &);
	inline void send_bits(TState &state, int value, int length);
	inline unsigned bi_reverse(unsigned code, int len);
	inline void bi_windup(TState &state);
	inline void copy_block(TState &state, char *buf, unsigned len, int header);


#define send_code(state, c, tree) send_bits(state, tree[c].fc.code, tree[c].dl.len)
	// Send a code of the given tree. c and tree must not have side effects

	// alternatively...
	//#define send_code(state, c, tree)
	//     { if (state.verbose>1) fprintf(stderr,"\ncd %3d ",(c));
	//       send_bits(state, tree[c].fc.code, tree[c].dl.len); }

#define d_code(dist) ((dist) < 256 ? state.ts.dist_code[dist] : state.ts.dist_code[256+((dist)>>7)])
	// Mapping from a distance to a distance code. dist is the distance - 1 and
	// must not have side effects. dist_code[256] and dist_code[257] are never used.

#define Max(a,b) (a >= b ? a : b)
	/* the arguments must not have side effects */

	/* ===========================================================================
	* Allocate the match buffer, initialize the various tables and save the
	* location of the internal file attribute (ascii/binary) and method
	* (DEFLATE/STORE).
	*/
	inline void ct_init(TState &state, ush *attr) {
		int n;        /* iterates over tree elements */
		int _bits;     /* bit counter */
		int length;   /* length value */
		int code;     /* code value */
		int dist;     /* distance index */

		state.ts.file_type = attr;
		//state.ts.file_method = method;
		state.ts.cmpr_bytelen = state.ts.cmpr_len_bits = 0L;
		state.ts.input_len = 0L;

		if (state.ts.static_dtree[0].dl.len != 0) return; /* ct_init already called */

														  /* Initialize the mapping length (0..255) -> length code (0..28) */
		length = 0;
		for (code = 0; code < LENGTH_CODES - 1; code++) {
			state.ts.base_length[code] = length;
			for (n = 0; n < (1 << extra_lbits[code]); n++) {
				state.ts.length_code[length++] = (uch)code;
			}
		}
		Assert(state, length == 256, "ct_init: length != 256");
		/* Note that the length 255 (match length 258) can be represented
		* in two different ways: code 284 + 5 bits or code 285, so we
		* overwrite length_code[255] to use the best encoding:
		*/
		state.ts.length_code[length - 1] = (uch)code;

		/* Initialize the mapping dist (0..32K) -> dist code (0..29) */
		dist = 0;
		for (code = 0; code < 16; code++) {
			state.ts.base_dist[code] = dist;
			for (n = 0; n < (1 << extra_dbits[code]); n++) {
				state.ts.dist_code[dist++] = (uch)code;
			}
		}
		Assert(state, dist == 256, "ct_init: dist != 256");
		dist >>= 7; /* from now on, all distances are divided by 128 */
		for (; code < D_CODES; code++) {
			state.ts.base_dist[code] = dist << 7;
			for (n = 0; n < (1 << (extra_dbits[code] - 7)); n++) {
				state.ts.dist_code[256 + dist++] = (uch)code;
			}
		}
		Assert(state, dist == 256, "ct_init: 256+dist != 512");

		/* Construct the codes of the static literal tree */
		for (_bits = 0; _bits <= MAX_BITS; _bits++) state.ts.bl_count[_bits] = 0;
		n = 0;
		while (n <= 143) state.ts.static_ltree[n++].dl.len = 8, state.ts.bl_count[8]++;
		while (n <= 255) state.ts.static_ltree[n++].dl.len = 9, state.ts.bl_count[9]++;
		while (n <= 279) state.ts.static_ltree[n++].dl.len = 7, state.ts.bl_count[7]++;
		while (n <= 287) state.ts.static_ltree[n++].dl.len = 8, state.ts.bl_count[8]++;
		/* fc.codes 286 and 287 do not exist, but we must include them in the
		* tree construction to get a canonical Huffman tree (longest code
		* all ones)
		*/
		gen_codes(state, (ct_data *)state.ts.static_ltree, L_CODES + 1);

		/* The static distance tree is trivial: */
		for (n = 0; n < D_CODES; n++) {
			state.ts.static_dtree[n].dl.len = 5;
			state.ts.static_dtree[n].fc.code = (ush)bi_reverse(n, 5);
		}

		/* Initialize the first block of the first file: */
		init_block(state);
	}

	/* ===========================================================================
	* Initialize a new block.
	*/
	inline void init_block(TState &state) {
		int n; /* iterates over tree elements */

			   /* Initialize the trees. */
		for (n = 0; n < L_CODES; n++) state.ts.dyn_ltree[n].fc.freq = 0;
		for (n = 0; n < D_CODES; n++) state.ts.dyn_dtree[n].fc.freq = 0;
		for (n = 0; n < BL_CODES; n++) state.ts.bl_tree[n].fc.freq = 0;

		state.ts.dyn_ltree[END_BLOCK].fc.freq = 1;
		state.ts.opt_len = state.ts.static_len = 0L;
		state.ts.last_lit = state.ts.last_dist = state.ts.last_flags = 0;
		state.ts.flags = 0; state.ts.flag_bit = 1;
	}

#define SMALLEST 1
	/* Index within the heap array of least frequent node in the Huffman tree */


	/* ===========================================================================
	* Remove the smallest element from the heap and recreate the heap with
	* one less element. Updates heap and heap_len.
	*/
#define pqremove(tree, top) \
{\
    top = state.ts.heap[SMALLEST]; \
    state.ts.heap[SMALLEST] = state.ts.heap[state.ts.heap_len--]; \
    pqdownheap(state,tree, SMALLEST); \
}

	/* ===========================================================================
	* Compares to subtrees, using the tree depth as tie breaker when
	* the subtrees have equal frequency. This minimizes the worst case length.
	*/
#define smaller(tree, n, m) \
   (tree[n].fc.freq < tree[m].fc.freq || \
   (tree[n].fc.freq == tree[m].fc.freq && state.ts.depth[n] <= state.ts.depth[m]))

	/* ===========================================================================
	* Restore the heap property by moving down the tree starting at node k,
	* exchanging a node with the smallest of its two sons if necessary, stopping
	* when the heap property is re-established (each father smaller than its
	* two sons).
	*/
	inline void pqdownheap(TState &state, ct_data *tree, int k) {
		int v = state.ts.heap[k];
		int j = k << 1;  /* left son of k */
		int htemp;       /* required because of bug in SASC compiler */

		while (j <= state.ts.heap_len) {
			/* Set j to the smallest of the two sons: */
			if (j < state.ts.heap_len && smaller(tree, state.ts.heap[j + 1], state.ts.heap[j])) j++;

			/* Exit if v is smaller than both sons */
			htemp = state.ts.heap[j];
			if (smaller(tree, v, htemp)) break;

			/* Exchange v with the smallest son */
			state.ts.heap[k] = htemp;
			k = j;

			/* And continue down the tree, setting j to the left son of k */
			j <<= 1;
		}
		state.ts.heap[k] = v;
	}

	/* ===========================================================================
	* Compute the optimal bit lengths for a tree and update the total bit length
	* for the current block.
	* IN assertion: the fields freq and dad are set, heap[heap_max] and
	*    above are the tree nodes sorted by increasing frequency.
	* OUT assertions: the field len is set to the optimal bit length, the
	*     array bl_count contains the frequencies for each bit length.
	*     The length opt_len is updated; static_len is also updated if stree is
	*     not null.
	*/
	inline void gen_bitlen(TState &state, tree_desc *desc) {
		ct_data *tree = desc->dyn_tree;
		const int *extra = desc->extra_bits;
		int base = desc->extra_base;
		int max_code = desc->max_code;
		int max_length = desc->max_length;
		ct_data *stree = desc->static_tree;
		int h;              /* heap index */
		int n, m;           /* iterate over the tree elements */
		int _bits;           /* bit length */
		int xbits;          /* extra bits */
		ush f;              /* frequency */
		int overflow = 0;   /* number of elements with bit length too large */

		for (_bits = 0; _bits <= MAX_BITS; _bits++) state.ts.bl_count[_bits] = 0;

		/* In a first pass, compute the optimal bit lengths (which may
		* overflow in the case of the bit length tree).
		*/
		tree[state.ts.heap[state.ts.heap_max]].dl.len = 0; /* root of the heap */

		for (h = state.ts.heap_max + 1; h < HEAP_SIZE; h++) {
			n = state.ts.heap[h];
			_bits = tree[tree[n].dl.dad].dl.len + 1;
			if (_bits > max_length) _bits = max_length, overflow++;
			tree[n].dl.len = (ush)_bits;
			/* We overwrite tree[n].dl.dad which is no longer needed */

			if (n > max_code) continue; /* not a leaf node */

			state.ts.bl_count[_bits]++;
			xbits = 0;
			if (n >= base) xbits = extra[n - base];
			f = tree[n].fc.freq;
			state.ts.opt_len += (ulg)f * (_bits + xbits);
			if (stree) state.ts.static_len += (ulg)f * (stree[n].dl.len + xbits);
		}
		if (overflow == 0) return;

		Trace("\nbit length overflow\n");
		/* This happens for example on obj2 and pic of the Calgary corpus */

		/* Find the first bit length which could increase: */
		do {
			_bits = max_length - 1;
			while (state.ts.bl_count[_bits] == 0) _bits--;
			state.ts.bl_count[_bits]--;           /* move one leaf down the tree */
			state.ts.bl_count[_bits + 1] += (ush)2; /* move one overflow item as its brother */
			state.ts.bl_count[max_length]--;
			/* The brother of the overflow item also moves one step up,
			* but this does not affect bl_count[max_length]
			*/
			overflow -= 2;
		} while (overflow > 0);

		/* Now recompute all bit lengths, scanning in increasing frequency.
		* h is still equal to HEAP_SIZE. (It is simpler to reconstruct all
		* lengths instead of fixing only the wrong ones. This idea is taken
		* from 'ar' written by Haruhiko Okumura.)
		*/
		for (_bits = max_length; _bits != 0; _bits--) {
			n = state.ts.bl_count[_bits];
			while (n != 0) {
				m = state.ts.heap[--h];
				if (m > max_code) continue;
				if (tree[m].dl.len != (ush)_bits) {
					Trace("code %d bits %d->%d\n", m, tree[m].dl.len, _bits);
					state.ts.opt_len += ((long)_bits - (long)tree[m].dl.len)*(long)tree[m].fc.freq;
					tree[m].dl.len = (ush)_bits;
				}
				n--;
			}
		}
	}

	/* ===========================================================================
	* Generate the codes for a given tree and bit counts (which need not be
	* optimal).
	* IN assertion: the array bl_count contains the bit length statistics for
	* the given tree and the field len is set for all tree elements.
	* OUT assertion: the field code is set for all tree elements of non
	*     zero code length.
	*/
	inline void gen_codes(TState &state, ct_data *tree, int max_code) {
		ush next_code[MAX_BITS + 1]; /* next code value for each bit length */
		ush code = 0;              /* running code value */
		int _bits;                  /* bit index */
		int n;                     /* code index */

								   /* The distribution counts are first used to generate the code values
								   * without bit reversal.
								   */
		for (_bits = 1; _bits <= MAX_BITS; _bits++) {
			next_code[_bits] = code = (ush)((code + state.ts.bl_count[_bits - 1]) << 1);
		}
		/* Check that the bit counts in bl_count are consistent. The last code
		* must be all ones.
		*/
		Assert(state, code + state.ts.bl_count[MAX_BITS] - 1 == (1 << ((ush)MAX_BITS)) - 1,
			   "inconsistent bit counts");
		Trace("\ngen_codes: max_code %d ", max_code);

		for (n = 0; n <= max_code; n++) {
			int len = tree[n].dl.len;
			if (len == 0) continue;
			/* Now reverse the bits */
			tree[n].fc.code = (ush)bi_reverse(next_code[len]++, len);

			//Tracec(tree != state.ts.static_ltree, "\nn %3d %c l %2d c %4x (%x) ", n, (isgraph(n) ? n : ' '), len, tree[n].fc.code, next_code[len]-1);
		}
	}

	/* ===========================================================================
	* Construct one Huffman tree and assigns the code bit strings and lengths.
	* Update the total bit length for the current block.
	* IN assertion: the field freq is set for all tree elements.
	* OUT assertions: the fields len and code are set to the optimal bit length
	*     and corresponding code. The length opt_len is updated; static_len is
	*     also updated if stree is not null. The field max_code is set.
	*/
	inline void build_tree(TState &state, tree_desc *desc) {
		ct_data *tree = desc->dyn_tree;
		ct_data *stree = desc->static_tree;
		int elems = desc->elems;
		int n, m;          /* iterate over heap elements */
		int max_code = -1; /* largest code with non zero frequency */
		int node = elems;  /* next internal node of the tree */

						   /* Construct the initial heap, with least frequent element in
						   * heap[SMALLEST]. The sons of heap[n] are heap[2*n] and heap[2*n+1].
						   * heap[0] is not used.
						   */
		state.ts.heap_len = 0, state.ts.heap_max = HEAP_SIZE;

		for (n = 0; n < elems; n++) {
			if (tree[n].fc.freq != 0) {
				state.ts.heap[++state.ts.heap_len] = max_code = n;
				state.ts.depth[n] = 0;
			} else {
				tree[n].dl.len = 0;
			}
		}

		/* The pkzip format requires that at least one distance code exists,
		* and that at least one bit should be sent even if there is only one
		* possible code. So to avoid special checks later on we force at least
		* two codes of non zero frequency.
		*/
		while (state.ts.heap_len < 2) {
			int newcp = state.ts.heap[++state.ts.heap_len] = (max_code < 2 ? ++max_code : 0);
			tree[newcp].fc.freq = 1;
			state.ts.depth[newcp] = 0;
			state.ts.opt_len--; if (stree) state.ts.static_len -= stree[newcp].dl.len;
			/* new is 0 or 1 so it does not have extra bits */
		}
		desc->max_code = max_code;

		/* The elements heap[heap_len/2+1 .. heap_len] are leaves of the tree,
		* establish sub-heaps of increasing lengths:
		*/
		for (n = state.ts.heap_len / 2; n >= 1; n--) pqdownheap(state, tree, n);

		/* Construct the Huffman tree by repeatedly combining the least two
		* frequent nodes.
		*/
		do {
			pqremove(tree, n);   /* n = node of least frequency */
			m = state.ts.heap[SMALLEST];  /* m = node of next least frequency */

			state.ts.heap[--state.ts.heap_max] = n; /* keep the nodes sorted by frequency */
			state.ts.heap[--state.ts.heap_max] = m;

			/* Create a new node father of n and m */
			tree[node].fc.freq = (ush)(tree[n].fc.freq + tree[m].fc.freq);
			state.ts.depth[node] = (uch)(Max(state.ts.depth[n], state.ts.depth[m]) + 1);
			tree[n].dl.dad = tree[m].dl.dad = (ush)node;
			/* and insert the new node in the heap */
			state.ts.heap[SMALLEST] = node++;
			pqdownheap(state, tree, SMALLEST);

		} while (state.ts.heap_len >= 2);

		state.ts.heap[--state.ts.heap_max] = state.ts.heap[SMALLEST];

		/* At this point, the fields freq and dad are set. We can now
		* generate the bit lengths.
		*/
		gen_bitlen(state, (tree_desc *)desc);

		/* The field len is now set, we can generate the bit codes */
		gen_codes(state, (ct_data *)tree, max_code);
	}

	/* ===========================================================================
	* Scan a literal or distance tree to determine the frequencies of the codes
	* in the bit length tree. Updates opt_len to take into account the repeat
	* counts. (The contribution of the bit length codes will be added later
	* during the construction of bl_tree.)
	*/
	inline void scan_tree(TState &state, ct_data *tree, int max_code) {
		int n;                     /* iterates over all tree elements */
		int prevlen = -1;          /* last emitted length */
		int curlen;                /* length of current code */
		int nextlen = tree[0].dl.len; /* length of next code */
		int count = 0;             /* repeat count of the current code */
		int max_count = 7;         /* max repeat count */
		int min_count = 4;         /* min repeat count */

		if (nextlen == 0) max_count = 138, min_count = 3;
		tree[max_code + 1].dl.len = (ush)-1; /* guard */

		for (n = 0; n <= max_code; n++) {
			curlen = nextlen; nextlen = tree[n + 1].dl.len;
			if (++count < max_count && curlen == nextlen) {
				continue;
			} else if (count < min_count) {
				state.ts.bl_tree[curlen].fc.freq = (ush)(state.ts.bl_tree[curlen].fc.freq + count);
			} else if (curlen != 0) {
				if (curlen != prevlen) state.ts.bl_tree[curlen].fc.freq++;
				state.ts.bl_tree[REP_3_6].fc.freq++;
			} else if (count <= 10) {
				state.ts.bl_tree[REPZ_3_10].fc.freq++;
			} else {
				state.ts.bl_tree[REPZ_11_138].fc.freq++;
			}
			count = 0; prevlen = curlen;
			if (nextlen == 0) {
				max_count = 138, min_count = 3;
			} else if (curlen == nextlen) {
				max_count = 6, min_count = 3;
			} else {
				max_count = 7, min_count = 4;
			}
		}
	}

	/* ===========================================================================
	* Send a literal or distance tree in compressed form, using the codes in
	* bl_tree.
	*/
	inline void send_tree(TState &state, ct_data *tree, int max_code) {
		int n;                     /* iterates over all tree elements */
		int prevlen = -1;          /* last emitted length */
		int curlen;                /* length of current code */
		int nextlen = tree[0].dl.len; /* length of next code */
		int count = 0;             /* repeat count of the current code */
		int max_count = 7;         /* max repeat count */
		int min_count = 4;         /* min repeat count */

								   /* tree[max_code+1].dl.len = -1; */  /* guard already set */
		if (nextlen == 0) max_count = 138, min_count = 3;

		for (n = 0; n <= max_code; n++) {
			curlen = nextlen; nextlen = tree[n + 1].dl.len;
			if (++count < max_count && curlen == nextlen) {
				continue;
			} else if (count < min_count) {
				do {
					send_code(state, curlen, state.ts.bl_tree);
				} while (--count != 0);

			} else if (curlen != 0) {
				if (curlen != prevlen) {
					send_code(state, curlen, state.ts.bl_tree); count--;
				}
				Assert(state, count >= 3 && count <= 6, " 3_6?");
				send_code(state, REP_3_6, state.ts.bl_tree); send_bits(state, count - 3, 2);

			} else if (count <= 10) {
				send_code(state, REPZ_3_10, state.ts.bl_tree); send_bits(state, count - 3, 3);

			} else {
				send_code(state, REPZ_11_138, state.ts.bl_tree); send_bits(state, count - 11, 7);
			}
			count = 0; prevlen = curlen;
			if (nextlen == 0) {
				max_count = 138, min_count = 3;
			} else if (curlen == nextlen) {
				max_count = 6, min_count = 3;
			} else {
				max_count = 7, min_count = 4;
			}
		}
	}

	/* ===========================================================================
	* Construct the Huffman tree for the bit lengths and return the index in
	* bl_order of the last bit length code to send.
	*/
	inline int build_bl_tree(TState &state) {
		int max_blindex;  /* index of last bit length code of non zero freq */

						  /* Determine the bit length frequencies for literal and distance trees */
		scan_tree(state, (ct_data *)state.ts.dyn_ltree, state.ts.l_desc.max_code);
		scan_tree(state, (ct_data *)state.ts.dyn_dtree, state.ts.d_desc.max_code);

		/* Build the bit length tree: */
		build_tree(state, (tree_desc *)(&state.ts.bl_desc));
		/* opt_len now includes the length of the tree representations, except
		* the lengths of the bit lengths codes and the 5+5+4 bits for the counts.
		*/

		/* Determine the number of bit length codes to send. The pkzip format
		* requires that at least 4 bit length codes be sent. (appnote.txt says
		* 3 but the actual value used is 4.)
		*/
		for (max_blindex = BL_CODES - 1; max_blindex >= 3; max_blindex--) {
			if (state.ts.bl_tree[bl_order[max_blindex]].dl.len != 0) break;
		}
		/* Update opt_len to include the bit length tree and counts */
		state.ts.opt_len += 3 * (max_blindex + 1) + 5 + 5 + 4;
		Trace("\ndyn trees: dyn %ld, stat %ld", state.ts.opt_len, state.ts.static_len);

		return max_blindex;
	}

	/* ===========================================================================
	* Send the header for a block using dynamic Huffman trees: the counts, the
	* lengths of the bit length codes, the literal tree and the distance tree.
	* IN assertion: lcodes >= 257, dcodes >= 1, blcodes >= 4.
	*/
	inline void send_all_trees(TState &state, int lcodes, int dcodes, int blcodes) {
		int rank;                    /* index in bl_order */

		Assert(state, lcodes >= 257 && dcodes >= 1 && blcodes >= 4, "not enough codes");
		Assert(state, lcodes <= L_CODES && dcodes <= D_CODES && blcodes <= BL_CODES,
			   "too many codes");
		Trace("\nbl counts: ");
		send_bits(state, lcodes - 257, 5);
		/* not +255 as stated in appnote.txt 1.93a or -256 in 2.04c */
		send_bits(state, dcodes - 1, 5);
		send_bits(state, blcodes - 4, 4); /* not -3 as stated in appnote.txt */
		for (rank = 0; rank < blcodes; rank++) {
			Trace("\nbl code %2d ", bl_order[rank]);
			send_bits(state, state.ts.bl_tree[bl_order[rank]].dl.len, 3);
		}
		Trace("\nbl tree: sent %ld", state.bs.bits_sent);

		send_tree(state, (ct_data *)state.ts.dyn_ltree, lcodes - 1); /* send the literal tree */
		Trace("\nlit tree: sent %ld", state.bs.bits_sent);

		send_tree(state, (ct_data *)state.ts.dyn_dtree, dcodes - 1); /* send the distance tree */
		Trace("\ndist tree: sent %ld", state.bs.bits_sent);
	}

	/* ===========================================================================
	* Determine the best encoding for the current block: dynamic trees, static
	* trees or store, and output the encoded block to the zip file. This function
	* returns the total compressed length (in bytes) for the file so far.
	*/
	inline ulg flush_block(TState &state, char *buf, ulg stored_len, int eof) {
		ulg opt_lenb, static_lenb; /* opt_len and static_len in bytes */
		int max_blindex;  /* index of last bit length code of non zero freq */

		state.ts.flag_buf[state.ts.last_flags] = state.ts.flags; /* Save the flags for the last 8 items */

																 /* Check if the file is ascii or binary */
		if (*state.ts.file_type == (ush)UNKNOWN) set_file_type(state);

		/* Construct the literal and distance trees */
		build_tree(state, (tree_desc *)(&state.ts.l_desc));
		Trace("\nlit data: dyn %ld, stat %ld", state.ts.opt_len, state.ts.static_len);

		build_tree(state, (tree_desc *)(&state.ts.d_desc));
		Trace("\ndist data: dyn %ld, stat %ld", state.ts.opt_len, state.ts.static_len);
		/* At this point, opt_len and static_len are the total bit lengths of
		* the compressed block data, excluding the tree representations.
		*/

		/* Build the bit length tree for the above two trees, and get the index
		* in bl_order of the last bit length code to send.
		*/
		max_blindex = build_bl_tree(state);

		/* Determine the best encoding. Compute first the block length in bytes */
		opt_lenb = (state.ts.opt_len + 3 + 7) >> 3;
		static_lenb = (state.ts.static_len + 3 + 7) >> 3;
		state.ts.input_len += stored_len; /* for debugging only */

		Trace("\nopt %lu(%lu) stat %lu(%lu) stored %lu lit %u dist %u ",
			  opt_lenb, state.ts.opt_len, static_lenb, state.ts.static_len, stored_len,
			  state.ts.last_lit, state.ts.last_dist);

		if (static_lenb <= opt_lenb) opt_lenb = static_lenb;

		// Originally, zip allowed the file to be transformed from a compressed
		// into a stored file in the case where compression failed, there
		// was only one block, and it was allowed to change. I've removed this
		// possibility since the code's cleaner if no changes are allowed.
		//if (stored_len <= opt_lenb && eof && state.ts.cmpr_bytelen == 0L
		//   && state.ts.cmpr_len_bits == 0L && state.seekable)
		//{   // && state.ts.file_method != NULL
		//    // Since LIT_BUFSIZE <= 2*WSIZE, the input data must be there:
		//    Assert(state,buf!=NULL,"block vanished");
		//    copy_block(state,buf, (unsigned)stored_len, 0); // without header
		//    state.ts.cmpr_bytelen = stored_len;
		//    Assert(state,false,"unimplemented *state.ts.file_method = STORE;");
		//    //*state.ts.file_method = STORE;
		//}
		//else
		if (stored_len + 4 <= opt_lenb && buf != (char*)NULL) {
			/* 4: two words for the lengths */
			/* The test buf != NULL is only necessary if LIT_BUFSIZE > WSIZE.
			* Otherwise we can't have processed more than WSIZE input bytes since
			* the last block flush, because compression would have been
			* successful. If LIT_BUFSIZE <= WSIZE, it is never too late to
			* transform a block into a stored block.
			*/
			send_bits(state, (STORED_BLOCK << 1) + eof, 3);  /* send block type */
			state.ts.cmpr_bytelen += ((state.ts.cmpr_len_bits + 3 + 7) >> 3) + stored_len + 4;
			state.ts.cmpr_len_bits = 0L;

			copy_block(state, buf, (unsigned)stored_len, 1); /* with header */
		} else if (static_lenb == opt_lenb) {
			send_bits(state, (STATIC_TREES << 1) + eof, 3);
			compress_block(state, (ct_data *)state.ts.static_ltree, (ct_data *)state.ts.static_dtree);
			state.ts.cmpr_len_bits += 3 + state.ts.static_len;
			state.ts.cmpr_bytelen += state.ts.cmpr_len_bits >> 3;
			state.ts.cmpr_len_bits &= 7L;
		} else {
			send_bits(state, (DYN_TREES << 1) + eof, 3);
			send_all_trees(state, state.ts.l_desc.max_code + 1, state.ts.d_desc.max_code + 1, max_blindex + 1);
			compress_block(state, (ct_data *)state.ts.dyn_ltree, (ct_data *)state.ts.dyn_dtree);
			state.ts.cmpr_len_bits += 3 + state.ts.opt_len;
			state.ts.cmpr_bytelen += state.ts.cmpr_len_bits >> 3;
			state.ts.cmpr_len_bits &= 7L;
		}
		Assert(state, ((state.ts.cmpr_bytelen << 3) + state.ts.cmpr_len_bits) == state.bs.bits_sent, "bad compressed size");
		init_block(state);

		if (eof) {
			// Assert(state,input_len == isize, "bad input size");
			bi_windup(state);
			state.ts.cmpr_len_bits += 7;  /* align on byte boundary */
		}
		Trace("\n");

		return state.ts.cmpr_bytelen + (state.ts.cmpr_len_bits >> 3);
	}

	/* ===========================================================================
	* Save the match info and tally the frequency counts. Return true if
	* the current block must be flushed.
	*/
	inline int ct_tally(TState &state, int dist, int lc) {
		state.ts.l_buf[state.ts.last_lit++] = (uch)lc;
		if (dist == 0) {
			/* lc is the unmatched char */
			state.ts.dyn_ltree[lc].fc.freq++;
		} else {
			/* Here, lc is the match length - MIN_MATCH */
			dist--;             /* dist = match distance - 1 */
			Assert(state, (ush)dist < (ush)_ZIP_MAX_DIST &&
				(ush)lc <= (ush)(MAX_MATCH - MIN_MATCH) &&
				   (ush)d_code(dist) < (ush)D_CODES, "ct_tally: bad match");

			state.ts.dyn_ltree[state.ts.length_code[lc] + LITERALS + 1].fc.freq++;
			state.ts.dyn_dtree[d_code(dist)].fc.freq++;

			state.ts.d_buf[state.ts.last_dist++] = (ush)dist;
			state.ts.flags |= state.ts.flag_bit;
		}
		state.ts.flag_bit <<= 1;

		/* Output the flags if they fill a byte: */
		if ((state.ts.last_lit & 7) == 0) {
			state.ts.flag_buf[state.ts.last_flags++] = state.ts.flags;
			state.ts.flags = 0, state.ts.flag_bit = 1;
		}
		/* Try to guess if it is profitable to stop the current block here */
		if (state.level > 2 && (state.ts.last_lit & 0xfff) == 0) {
			/* Compute an upper bound for the compressed length */
			ulg out_length = (ulg)state.ts.last_lit * 8L;
			ulg in_length = (ulg)state.ds.strstart - state.ds.block_start;
			int dcode;
			for (dcode = 0; dcode < D_CODES; dcode++) {
				out_length += (ulg)state.ts.dyn_dtree[dcode].fc.freq*(5L + extra_dbits[dcode]);
			}
			out_length >>= 3;
			Trace("\nlast_lit %u, last_dist %u, in %ld, out ~%ld(%ld%%) ",
				  state.ts.last_lit, state.ts.last_dist, in_length, out_length,
				  100L - out_length * 100L / in_length);
			if (state.ts.last_dist < state.ts.last_lit / 2 && out_length < in_length / 2) return 1;
		}
		return (state.ts.last_lit == LIT_BUFSIZE - 1 || state.ts.last_dist == DIST_BUFSIZE);
		/* We avoid equality with LIT_BUFSIZE because of wraparound at 64K
		* on 16 bit machines and because stored blocks are restricted to
		* 64K-1 bytes.
		*/
	}

	/* ===========================================================================
	* Send the block data compressed using the given Huffman trees
	*/
	inline void compress_block(TState &state, ct_data *ltree, ct_data *dtree) {
		unsigned dist;      /* distance of matched string */
		int lc;             /* match length or unmatched char (if dist == 0) */
		unsigned lx = 0;    /* running index in l_buf */
		unsigned dx = 0;    /* running index in d_buf */
		unsigned fx = 0;    /* running index in flag_buf */
		uch flag = 0;       /* current flags */
		unsigned code;      /* the code to send */
		int extra;          /* number of extra bits to send */

		if (state.ts.last_lit != 0) do {
			if ((lx & 7) == 0) flag = state.ts.flag_buf[fx++];
			lc = state.ts.l_buf[lx++];
			if ((flag & 1) == 0) {
				send_code(state, lc, ltree); /* send a literal byte */
			} else {
				/* Here, lc is the match length - MIN_MATCH */
				code = state.ts.length_code[lc];
				send_code(state, code + LITERALS + 1, ltree); /* send the length code */
				extra = extra_lbits[code];
				if (extra != 0) {
					lc -= state.ts.base_length[code];
					send_bits(state, lc, extra);        /* send the extra length bits */
				}
				dist = state.ts.d_buf[dx++];
				/* Here, dist is the match distance - 1 */
				code = d_code(dist);
				Assert(state, code < D_CODES, "bad d_code");

				send_code(state, code, dtree);       /* send the distance code */
				extra = extra_dbits[code];
				if (extra != 0) {
					dist -= state.ts.base_dist[code];
					send_bits(state, dist, extra);   /* send the extra distance bits */
				}
			} /* literal or match pair ? */
			flag >>= 1;
		} while (lx < state.ts.last_lit);

		send_code(state, END_BLOCK, ltree);
	}

	/* ===========================================================================
	* Set the file type to ASCII or BINARY, using a crude approximation:
	* binary if more than 20% of the bytes are <= 6 or >= 128, ascii otherwise.
	* IN assertion: the fields freq of dyn_ltree are set and the total of all
	* frequencies does not exceed 64K (to fit in an int on 16 bit machines).
	*/
	inline void set_file_type(TState &state) {
		int n = 0;
		unsigned ascii_freq = 0;
		unsigned bin_freq = 0;
		while (n < 7)        bin_freq += state.ts.dyn_ltree[n++].fc.freq;
		while (n < 128)    ascii_freq += state.ts.dyn_ltree[n++].fc.freq;
		while (n < LITERALS) bin_freq += state.ts.dyn_ltree[n++].fc.freq;
		*state.ts.file_type = (ush)(bin_freq >(ascii_freq >> 2) ? BINARY : ASCII);
	}


	/* ===========================================================================
	* Initialize the bit string routines.
	*/
	inline void bi_init(TState &state, char *tgt_buf, unsigned tgt_size, int flsh_allowed) {
		state.bs.out_buf = tgt_buf;
		state.bs.out_size = tgt_size;
		state.bs.out_offset = 0;
		state.bs.flush_flg = flsh_allowed;

		state.bs.bi_buf = 0;
		state.bs.bi_valid = 0;
		state.bs.bits_sent = 0L;
	}

	/* ===========================================================================
	* Send a value on a given number of bits.
	* IN assertion: length <= 16 and value fits in length bits.
	*/
	inline void send_bits(TState &state, int value, int length) {
		Assert(state, length > 0 && length <= 15, "invalid length");
		state.bs.bits_sent += (ulg)length;
		/* If not enough room in bi_buf, use (bi_valid) bits from bi_buf and
		* (Buf_size - bi_valid) bits from value to flush the filled bi_buf,
		* then fill in the rest of (value), leaving (length - (Buf_size-bi_valid))
		* unused bits in bi_buf.
		*/
		state.bs.bi_buf |= (value << state.bs.bi_valid);
		state.bs.bi_valid += length;
		if (state.bs.bi_valid > (int)Buf_size) {
			PUTSHORT(state, state.bs.bi_buf);
			state.bs.bi_valid -= Buf_size;
			state.bs.bi_buf = (unsigned)value >> (length - state.bs.bi_valid);
		}
	}

	/* ===========================================================================
	* Reverse the first len bits of a code, using straightforward code (a faster
	* method would use a table)
	* IN assertion: 1 <= len <= 15
	*/
	inline unsigned bi_reverse(unsigned code, int len) {
		register unsigned res = 0;
		do {
			res |= code & 1;
			code >>= 1, res <<= 1;
		} while (--len > 0);
		return res >> 1;
	}

	/* ===========================================================================
	* Write out any remaining bits in an incomplete byte.
	*/
	inline void bi_windup(TState &state) {
		if (state.bs.bi_valid > 8) {
			PUTSHORT(state, state.bs.bi_buf);
		} else if (state.bs.bi_valid > 0) {
			PUTBYTE(state, state.bs.bi_buf);
		}
		if (state.bs.flush_flg) {
			state.flush_outbuf(state.param, state.bs.out_buf, &state.bs.out_offset);
		}
		state.bs.bi_buf = 0;
		state.bs.bi_valid = 0;
		state.bs.bits_sent = (state.bs.bits_sent + 7) & ~7;
	}

	/* ===========================================================================
	* Copy a stored block to the zip file, storing first the length and its
	* one's complement if requested.
	*/
	inline void copy_block(TState &state, char *block, unsigned len, int header) {
		bi_windup(state);              /* align on byte boundary */

		if (header) {
			PUTSHORT(state, (ush)len);
			PUTSHORT(state, (ush)~len);
			state.bs.bits_sent += 2 * 16;
		}
		if (state.bs.flush_flg) {
			state.flush_outbuf(state.param, state.bs.out_buf, &state.bs.out_offset);
			state.bs.out_offset = len;
			state.flush_outbuf(state.param, block, &state.bs.out_offset);
		} else if (state.bs.out_offset + len > state.bs.out_size) {
			Assert(state, false, "output buffer too small for in-memory compression");
		} else {
			memcpy(state.bs.out_buf + state.bs.out_offset, block, len);
			state.bs.out_offset += len;
		}
		state.bs.bits_sent += (ulg)len << 3;
	}








	/* ===========================================================================
	*  Prototypes for functions.
	*/

	inline void fill_window(TState &state);
	inline ulg deflate_fast(TState &state);

	inline int  longest_match(TState &state, IPos cur_match);


	/* ===========================================================================
	* Update a hash value with the given input byte
	* IN  assertion: all calls to to UPDATE_HASH are made with consecutive
	*    input characters, so that a running hash key can be computed from the
	*    previous key instead of complete recalculation each time.
	*/
#define UPDATE_HASH(h,c) (h = (((h)<<H_SHIFT) ^ (c)) & HASH_MASK)

	/* ===========================================================================
	* Insert string s in the dictionary and set match_head to the previous head
	* of the hash chain (the most recent string with same hash key). Return
	* the previous length of the hash chain.
	* IN  assertion: all calls to to INSERT_STRING are made with consecutive
	*    input characters and the first MIN_MATCH bytes of s are valid
	*    (except for the last MIN_MATCH-1 bytes of the input file).
	*/
#define INSERT_STRING(s, match_head) \
   (UPDATE_HASH(state.ds.ins_h, state.ds.window[(s) + (MIN_MATCH-1)]), \
    state.ds.prev[(s) & WMASK] = match_head = state.ds.head[state.ds.ins_h], \
    state.ds.head[state.ds.ins_h] = (s))

	/* ===========================================================================
	* Initialize the "longest match" routines for a new file
	*
	* IN assertion: window_size is > 0 if the input file is already read or
	*    mmap'ed in the window[] array, 0 otherwise. In the first case,
	*    window_size is sufficient to contain the whole input file plus
	*    MIN_LOOKAHEAD bytes (to avoid referencing memory beyond the end
	*    of window[] when looking for matches towards the end).
	*/
	inline void lm_init(TState &state, int pack_level, ush *flags) {
		register unsigned j;

		Assert(state, pack_level >= 1 && pack_level <= 8, "bad pack level");

		/* Do not slide the window if the whole input is already in memory
		* (window_size > 0)
		*/
		state.ds.sliding = 0;
		if (state.ds.window_size == 0L) {
			state.ds.sliding = 1;
			state.ds.window_size = (ulg)2L * WSIZE;
		}

		/* Initialize the hash table (avoiding 64K overflow for 16 bit systems).
		* prev[] will be initialized on the fly.
		*/
		state.ds.head[HASH_SIZE - 1] = Z_NIL;
		memset((char*)state.ds.head, Z_NIL, (unsigned)(HASH_SIZE - 1) * sizeof(*state.ds.head));

		/* Set the default configuration parameters:
		*/
		state.ds.max_lazy_match = configuration_table[pack_level].max_lazy;
		state.ds.good_match = configuration_table[pack_level].good_length;
		state.ds.nice_match = configuration_table[pack_level].nice_length;
		state.ds.max_chain_length = configuration_table[pack_level].max_chain;
		if (pack_level <= 2) {
			*flags |= Z_FAST;
		} else if (pack_level >= 8) {
			*flags |= Z_SLOW;
		}
		/* ??? reduce max_chain_length for binary files */

		state.ds.strstart = 0;
		state.ds.block_start = 0L;

		j = WSIZE;
		j <<= 1; // Can read 64K in one step
		state.ds.lookahead = state.readfunc(state, (char*)state.ds.window, j);

		if (state.ds.lookahead == 0 || state.ds.lookahead == (unsigned)EOF) {
			state.ds.eofile = 1, state.ds.lookahead = 0;
			return;
		}
		state.ds.eofile = 0;
		/* Make sure that we always have enough lookahead. This is important
		* if input comes from a device such as a tty.
		*/
		if (state.ds.lookahead < MIN_LOOKAHEAD) fill_window(state);

		state.ds.ins_h = 0;
		for (j = 0; j < MIN_MATCH - 1; j++) UPDATE_HASH(state.ds.ins_h, state.ds.window[j]);
		/* If lookahead < MIN_MATCH, ins_h is garbage, but this is
		* not important since only literal bytes will be emitted.
		*/
	}


	/* ===========================================================================
	* Set match_start to the longest match starting at the given string and
	* return its length. Matches shorter or equal to prev_length are discarded,
	* in which case the result is equal to prev_length and match_start is
	* garbage.
	* IN assertions: cur_match is the head of the hash chain for the current
	*   string (strstart) and its distance is <= _ZIP_MAX_DIST, and prev_length >= 1
	*/
	// For 80x86 and 680x0 and ARM, an optimized version is in match.asm or
	// match.S. The code is functionally equivalent, so you can use the C version
	// if desired. Which I do so desire!
	inline int longest_match(TState &state, IPos cur_match) {
		unsigned chain_length = state.ds.max_chain_length;   /* max hash chain length */
		register uch *scan = state.ds.window + state.ds.strstart; /* current string */
		register uch *match;                    /* matched string */
		register int len;                           /* length of current match */
		int best_len = state.ds.prev_length;                 /* best match length so far */
		IPos limit = state.ds.strstart > (IPos)_ZIP_MAX_DIST ? state.ds.strstart - (IPos)_ZIP_MAX_DIST : Z_NIL;
		/* Stop when cur_match becomes <= limit. To simplify the code,
		* we prevent matches with the string of window index 0.
		*/

		// The code is optimized for HASH_BITS >= 8 and MAX_MATCH-2 multiple of 16.
		// It is easy to get rid of this optimization if necessary.
		Assert(state, HASH_BITS >= 8 && MAX_MATCH == 258, "Code too clever");



		register uch *strend = state.ds.window + state.ds.strstart + MAX_MATCH;
		register uch scan_end1 = scan[best_len - 1];
		register uch scan_end = scan[best_len];

		/* Do not waste too much time if we already have a good match: */
		if (state.ds.prev_length >= state.ds.good_match) {
			chain_length >>= 2;
		}

		Assert(state, state.ds.strstart <= state.ds.window_size - MIN_LOOKAHEAD, "insufficient lookahead");

		do {
			Assert(state, cur_match < state.ds.strstart, "no future");
			match = state.ds.window + cur_match;

			/* Skip to next match if the match length cannot increase
			* or if the match length is less than 2:
			*/
			if (match[best_len] != scan_end ||
				match[best_len - 1] != scan_end1 ||
				*match != *scan ||
				*++match != scan[1])      continue;

			/* The check at best_len-1 can be removed because it will be made
			* again later. (This heuristic is not always a win.)
			* It is not necessary to compare scan[2] and match[2] since they
			* are always equal when the other bytes match, given that
			* the hash keys are equal and that HASH_BITS >= 8.
			*/
			scan += 2, match++;

			/* We check for insufficient lookahead only every 8th comparison;
			* the 256th check will be made at strstart+258.
			*/
			do {
			} while (*++scan == *++match && *++scan == *++match &&
					 *++scan == *++match && *++scan == *++match &&
					 *++scan == *++match && *++scan == *++match &&
					 *++scan == *++match && *++scan == *++match &&
					 scan < strend);

			Assert(state, scan <= state.ds.window + (unsigned)(state.ds.window_size - 1), "wild scan");

			len = MAX_MATCH - (int)(strend - scan);
			scan = strend - MAX_MATCH;


			if (len > best_len) {
				state.ds.match_start = cur_match;
				best_len = len;
				if (len >= state.ds.nice_match) break;
				scan_end1 = scan[best_len - 1];
				scan_end = scan[best_len];
			}
		} while ((cur_match = state.ds.prev[cur_match & WMASK]) > limit
				 && --chain_length != 0);

		return best_len;
	}



#define check_match(state,start, match, length)
	// or alternatively...
	//void check_match(TState &state,IPos start, IPos match, int length)
	//{ // check that the match is indeed a match
	//    if (memcmp((char*)state.ds.window + match,
	//                (char*)state.ds.window + start, length) != EQUAL) {
	//        fprintf(stderr,
	//            " start %d, match %d, length %d\n",
	//            start, match, length);
	//        error("invalid match");
	//    }
	//    if (state.verbose > 1) {
	//        fprintf(stderr,"\\[%d,%d]", start-match, length);
	//        do { fprintf(stdout,"%c",state.ds.window[start++]); } while (--length != 0);
	//    }
	//}

	/* ===========================================================================
	* Fill the window when the lookahead becomes insufficient.
	* Updates strstart and lookahead, and sets eofile if end of input file.
	*
	* IN assertion: lookahead < MIN_LOOKAHEAD && strstart + lookahead > 0
	* OUT assertions: strstart <= window_size-MIN_LOOKAHEAD
	*    At least one byte has been read, or eofile is set; file reads are
	*    performed for at least two bytes (required for the translate_eol option).
	*/
	inline void fill_window(TState &state) {
		register unsigned n, m;
		unsigned more;    /* Amount of free space at the end of the window. */

		do {
			more = (unsigned)(state.ds.window_size - (ulg)state.ds.lookahead - (ulg)state.ds.strstart);

			/* If the window is almost full and there is insufficient lookahead,
			* move the upper half to the lower one to make room in the upper half.
			*/
			if (more == (unsigned)EOF) {
				/* Very unlikely, but possible on 16 bit machine if strstart == 0
				* and lookahead == 1 (input done one byte at time)
				*/
				more--;

				/* For MMAP or BIG_MEM, the whole input file is already in memory so
				* we must not perform sliding. We must however call (*read_buf)() in
				* order to compute the crc, update lookahead and possibly set eofile.
				*/
			} else if (state.ds.strstart >= WSIZE + _ZIP_MAX_DIST && state.ds.sliding) {

				/* By the IN assertion, the window is not empty so we can't confuse
				* more == 0 with more == 64K on a 16 bit machine.
				*/
				memcpy((char*)state.ds.window, (char*)state.ds.window + WSIZE, (unsigned)WSIZE);
				state.ds.match_start -= WSIZE;
				state.ds.strstart -= WSIZE; /* we now have strstart >= _ZIP_MAX_DIST: */

				state.ds.block_start -= (long)WSIZE;

				for (n = 0; n < HASH_SIZE; n++) {
					m = state.ds.head[n];
					state.ds.head[n] = (Pos)(m >= WSIZE ? m - WSIZE : Z_NIL);
				}
				for (n = 0; n < WSIZE; n++) {
					m = state.ds.prev[n];
					state.ds.prev[n] = (Pos)(m >= WSIZE ? m - WSIZE : Z_NIL);
					/* If n is not on any hash chain, prev[n] is garbage but
					* its value will never be used.
					*/
				}
				more += WSIZE;
			}
			if (state.ds.eofile) return;

			/* If there was no sliding:
			*    strstart <= WSIZE+_ZIP_MAX_DIST-1 && lookahead <= MIN_LOOKAHEAD - 1 &&
			*    more == window_size - lookahead - strstart
			* => more >= window_size - (MIN_LOOKAHEAD-1 + WSIZE + _ZIP_MAX_DIST-1)
			* => more >= window_size - 2*WSIZE + 2
			* In the MMAP or BIG_MEM case (not yet supported in gzip),
			*   window_size == input_size + MIN_LOOKAHEAD  &&
			*   strstart + lookahead <= input_size => more >= MIN_LOOKAHEAD.
			* Otherwise, window_size == 2*WSIZE so more >= 2.
			* If there was sliding, more >= WSIZE. So in all cases, more >= 2.
			*/
			Assert(state, more >= 2, "more < 2");

			n = state.readfunc(state, (char*)state.ds.window + state.ds.strstart + state.ds.lookahead, more);

			if (n == 0 || n == (unsigned)EOF) {
				state.ds.eofile = 1;
			} else {
				state.ds.lookahead += n;
			}
		} while (state.ds.lookahead < MIN_LOOKAHEAD && !state.ds.eofile);
	}

	/* ===========================================================================
	* Flush the current block, with given end-of-file flag.
	* IN assertion: strstart is set to the end of the current match.
	*/
#define FLUSH_BLOCK(state,eof) \
   flush_block(state,state.ds.block_start >= 0L ? (char*)&state.ds.window[(unsigned)state.ds.block_start] : \
                (char*)NULL, (long)state.ds.strstart - state.ds.block_start, (eof))

	/* ===========================================================================
	* Processes a new input file and return its compressed length. This
	* function does not perform lazy evaluation of matches and inserts
	* new strings in the dictionary only for unmatched strings or for short
	* matches. It is used only for the fast compression options.
	*/
	inline ulg deflate_fast(TState &state) {
		IPos hash_head = Z_NIL;       /* head of the hash chain */
		int flush;                  /* set if current block must be flushed */
		unsigned match_length = 0;  /* length of best match */

		state.ds.prev_length = MIN_MATCH - 1;
		while (state.ds.lookahead != 0) {
			/* Insert the string window[strstart .. strstart+2] in the
			* dictionary, and set hash_head to the head of the hash chain:
			*/
			if (state.ds.lookahead >= MIN_MATCH)
				INSERT_STRING(state.ds.strstart, hash_head);

			/* Find the longest match, discarding those <= prev_length.
			* At this point we have always match_length < MIN_MATCH
			*/
			if (hash_head != Z_NIL && state.ds.strstart - hash_head <= _ZIP_MAX_DIST) {
				/* To simplify the code, we prevent matches with the string
				* of window index 0 (in particular we have to avoid a match
				* of the string with itself at the start of the input file).
				*/
				/* Do not look for matches beyond the end of the input.
				* This is necessary to make deflate deterministic.
				*/
				if ((unsigned)state.ds.nice_match > state.ds.lookahead) state.ds.nice_match = (int)state.ds.lookahead;
				match_length = longest_match(state, hash_head);
				/* longest_match() sets match_start */
				if (match_length > state.ds.lookahead) match_length = state.ds.lookahead;
			}
			if (match_length >= MIN_MATCH) {
				check_match(state, state.ds.strstart, state.ds.match_start, match_length);

				flush = ct_tally(state, state.ds.strstart - state.ds.match_start, match_length - MIN_MATCH);

				state.ds.lookahead -= match_length;

				/* Insert new strings in the hash table only if the match length
				* is not too large. This saves time but degrades compression.
				*/
				if (match_length <= state.ds.max_insert_length
					&& state.ds.lookahead >= MIN_MATCH) {
					match_length--; /* string at strstart already in hash table */
					do {
						state.ds.strstart++;
						INSERT_STRING(state.ds.strstart, hash_head);
						/* strstart never exceeds WSIZE-MAX_MATCH, so there are
						* always MIN_MATCH bytes ahead.
						*/
					} while (--match_length != 0);
					state.ds.strstart++;
				} else {
					state.ds.strstart += match_length;
					match_length = 0;
					state.ds.ins_h = state.ds.window[state.ds.strstart];
					UPDATE_HASH(state.ds.ins_h, state.ds.window[state.ds.strstart + 1]);
					Assert(state, MIN_MATCH == 3, "Call UPDATE_HASH() MIN_MATCH-3 more times");
				}
			} else {
				/* No match, output a literal byte */
				flush = ct_tally(state, 0, state.ds.window[state.ds.strstart]);
				state.ds.lookahead--;
				state.ds.strstart++;
			}
			if (flush) FLUSH_BLOCK(state, 0), state.ds.block_start = state.ds.strstart;

			/* Make sure that we always have enough lookahead, except
			* at the end of the input file. We need MAX_MATCH bytes
			* for the next match, plus MIN_MATCH bytes to insert the
			* string following the next match.
			*/
			if (state.ds.lookahead < MIN_LOOKAHEAD) fill_window(state);
		}
		return FLUSH_BLOCK(state, 1); /* eof */
	}

	/* ===========================================================================
	* Same as above, but achieves better compression. We use a lazy
	* evaluation for matches: a match is finally adopted only if there is
	* no better match at the next window position.
	*/
	inline ulg deflate(TState &state) {
		IPos hash_head = Z_NIL;       /* head of hash chain */
		IPos prev_match;            /* previous match */
		int flush;                  /* set if current block must be flushed */
		int match_available = 0;    /* set if previous match exists */
		register unsigned match_length = MIN_MATCH - 1; /* length of best match */

		if (state.level <= 3) return deflate_fast(state); /* optimized for speed */

														  /* Process the input block. */
		while (state.ds.lookahead != 0) {
			/* Insert the string window[strstart .. strstart+2] in the
			* dictionary, and set hash_head to the head of the hash chain:
			*/
			if (state.ds.lookahead >= MIN_MATCH)
				INSERT_STRING(state.ds.strstart, hash_head);

			/* Find the longest match, discarding those <= prev_length.
			*/
			state.ds.prev_length = match_length, prev_match = state.ds.match_start;
			match_length = MIN_MATCH - 1;

			if (hash_head != Z_NIL && state.ds.prev_length < state.ds.max_lazy_match &&
				state.ds.strstart - hash_head <= _ZIP_MAX_DIST) {
				/* To simplify the code, we prevent matches with the string
				* of window index 0 (in particular we have to avoid a match
				* of the string with itself at the start of the input file).
				*/
				/* Do not look for matches beyond the end of the input.
				* This is necessary to make deflate deterministic.
				*/
				if ((unsigned)state.ds.nice_match > state.ds.lookahead) state.ds.nice_match = (int)state.ds.lookahead;
				match_length = longest_match(state, hash_head);
				/* longest_match() sets match_start */
				if (match_length > state.ds.lookahead) match_length = state.ds.lookahead;

				/* Ignore a length 3 match if it is too distant: */
				if (match_length == MIN_MATCH && state.ds.strstart - state.ds.match_start > Z_TOO_FAR) {
					/* If prev_match is also MIN_MATCH, match_start is garbage
					* but we will ignore the current match anyway.
					*/
					match_length = MIN_MATCH - 1;
				}
			}
			/* If there was a match at the previous step and the current
			* match is not better, output the previous match:
			*/
			if (state.ds.prev_length >= MIN_MATCH && match_length <= state.ds.prev_length) {
				unsigned max_insert = state.ds.strstart + state.ds.lookahead - MIN_MATCH;
				check_match(state, state.ds.strstart - 1, prev_match, state.ds.prev_length);
				flush = ct_tally(state, state.ds.strstart - 1 - prev_match, state.ds.prev_length - MIN_MATCH);

				/* Insert in hash table all strings up to the end of the match.
				* strstart-1 and strstart are already inserted.
				*/
				state.ds.lookahead -= state.ds.prev_length - 1;
				state.ds.prev_length -= 2;
				do {
					if (++state.ds.strstart <= max_insert) {
						INSERT_STRING(state.ds.strstart, hash_head);
						/* strstart never exceeds WSIZE-MAX_MATCH, so there are
						* always MIN_MATCH bytes ahead.
						*/
					}
				} while (--state.ds.prev_length != 0);
				state.ds.strstart++;
				match_available = 0;
				match_length = MIN_MATCH - 1;

				if (flush) FLUSH_BLOCK(state, 0), state.ds.block_start = state.ds.strstart;

			} else if (match_available) {
				/* If there was no match at the previous position, output a
				* single literal. If there was a match but the current match
				* is longer, truncate the previous match to a single literal.
				*/
				if (ct_tally(state, 0, state.ds.window[state.ds.strstart - 1])) {
					FLUSH_BLOCK(state, 0), state.ds.block_start = state.ds.strstart;
				}
				state.ds.strstart++;
				state.ds.lookahead--;
			} else {
				/* There is no previous match to compare with, wait for
				* the next step to decide.
				*/
				match_available = 1;
				state.ds.strstart++;
				state.ds.lookahead--;
			}
			//        Assert(state,strstart <= isize && lookahead <= isize, "a bit too far");

			/* Make sure that we always have enough lookahead, except
			* at the end of the input file. We need MAX_MATCH bytes
			* for the next match, plus MIN_MATCH bytes to insert the
			* string following the next match.
			*/
			if (state.ds.lookahead < MIN_LOOKAHEAD) fill_window(state);
		}
		if (match_available) ct_tally(state, 0, state.ds.window[state.ds.strstart - 1]);

		return FLUSH_BLOCK(state, 1); /* eof */
	}












	inline int putlocal(struct zlist *z, WRITEFUNC wfunc, void *param) { // Write a local header described by *z to file *f.  Return a ZE_ error code.
		PUTLG(LOCSIG, f);
		PUTSH(z->ver, f);
		PUTSH(z->lflg, f);
		PUTSH(z->how, f);
		PUTLG(z->tim, f);
		PUTLG(z->crc, f);
		PUTLG(z->siz, f);
		PUTLG(z->len, f);
		PUTSH(z->nam, f);
		PUTSH(z->ext, f);
		size_t res = (size_t)wfunc(param, z->iname, (unsigned int)z->nam);
		if (res != z->nam) return ZE_TEMP;
		if (z->ext) {
			res = (size_t)wfunc(param, z->extra, (unsigned int)z->ext);
			if (res != z->ext) return ZE_TEMP;
		}
		return ZE_OK;
	}

	inline int putextended(struct zlist *z, WRITEFUNC wfunc, void *param) { // Write an extended local header described by *z to file *f. Returns a ZE_ code
		PUTLG(EXTLOCSIG, f);
		PUTLG(z->crc, f);
		PUTLG(z->siz, f);
		PUTLG(z->len, f);
		return ZE_OK;
	}

	inline int putcentral(struct zlist *z, WRITEFUNC wfunc, void *param) { // Write a central header entry of *z to file *f. Returns a ZE_ code.
		PUTLG(CENSIG, f);
		PUTSH(z->vem, f);
		PUTSH(z->ver, f);
		PUTSH(z->flg, f);
		PUTSH(z->how, f);
		PUTLG(z->tim, f);
		PUTLG(z->crc, f);
		PUTLG(z->siz, f);
		PUTLG(z->len, f);
		PUTSH(z->nam, f);
		PUTSH(z->cext, f);
		PUTSH(z->com, f);
		PUTSH(z->dsk, f);
		PUTSH(z->att, f);
		PUTLG(z->atx, f);
		PUTLG(z->off, f);
		if ((size_t)wfunc(param, z->iname, (unsigned int)z->nam) != z->nam ||
			(z->cext && (size_t)wfunc(param, z->cextra, (unsigned int)z->cext) != z->cext) ||
			(z->com && (size_t)wfunc(param, z->comment, (unsigned int)z->com) != z->com))
			return ZE_TEMP;
		return ZE_OK;
	}


	inline int putend(int n, ulg s, ulg c, extent m, char *z, WRITEFUNC wfunc, void *param) { // write the end of the central-directory-data to file *f.
		PUTLG(ENDSIG, f);
		PUTSH(0, f);
		PUTSH(0, f);
		PUTSH(n, f);
		PUTSH(n, f);
		PUTLG(s, f);
		PUTLG(c, f);
		PUTSH(m, f);
		// Write the comment, if any
		if (m && wfunc(param, z, (unsigned int)m) != m) return ZE_TEMP;
		return ZE_OK;
	}





#ifndef _unzip_H
	const ulg crc_table[256] = {
		0x00000000L, 0x77073096L, 0xee0e612cL, 0x990951baL, 0x076dc419L,
		0x706af48fL, 0xe963a535L, 0x9e6495a3L, 0x0edb8832L, 0x79dcb8a4L,
		0xe0d5e91eL, 0x97d2d988L, 0x09b64c2bL, 0x7eb17cbdL, 0xe7b82d07L,
		0x90bf1d91L, 0x1db71064L, 0x6ab020f2L, 0xf3b97148L, 0x84be41deL,
		0x1adad47dL, 0x6ddde4ebL, 0xf4d4b551L, 0x83d385c7L, 0x136c9856L,
		0x646ba8c0L, 0xfd62f97aL, 0x8a65c9ecL, 0x14015c4fL, 0x63066cd9L,
		0xfa0f3d63L, 0x8d080df5L, 0x3b6e20c8L, 0x4c69105eL, 0xd56041e4L,
		0xa2677172L, 0x3c03e4d1L, 0x4b04d447L, 0xd20d85fdL, 0xa50ab56bL,
		0x35b5a8faL, 0x42b2986cL, 0xdbbbc9d6L, 0xacbcf940L, 0x32d86ce3L,
		0x45df5c75L, 0xdcd60dcfL, 0xabd13d59L, 0x26d930acL, 0x51de003aL,
		0xc8d75180L, 0xbfd06116L, 0x21b4f4b5L, 0x56b3c423L, 0xcfba9599L,
		0xb8bda50fL, 0x2802b89eL, 0x5f058808L, 0xc60cd9b2L, 0xb10be924L,
		0x2f6f7c87L, 0x58684c11L, 0xc1611dabL, 0xb6662d3dL, 0x76dc4190L,
		0x01db7106L, 0x98d220bcL, 0xefd5102aL, 0x71b18589L, 0x06b6b51fL,
		0x9fbfe4a5L, 0xe8b8d433L, 0x7807c9a2L, 0x0f00f934L, 0x9609a88eL,
		0xe10e9818L, 0x7f6a0dbbL, 0x086d3d2dL, 0x91646c97L, 0xe6635c01L,
		0x6b6b51f4L, 0x1c6c6162L, 0x856530d8L, 0xf262004eL, 0x6c0695edL,
		0x1b01a57bL, 0x8208f4c1L, 0xf50fc457L, 0x65b0d9c6L, 0x12b7e950L,
		0x8bbeb8eaL, 0xfcb9887cL, 0x62dd1ddfL, 0x15da2d49L, 0x8cd37cf3L,
		0xfbd44c65L, 0x4db26158L, 0x3ab551ceL, 0xa3bc0074L, 0xd4bb30e2L,
		0x4adfa541L, 0x3dd895d7L, 0xa4d1c46dL, 0xd3d6f4fbL, 0x4369e96aL,
		0x346ed9fcL, 0xad678846L, 0xda60b8d0L, 0x44042d73L, 0x33031de5L,
		0xaa0a4c5fL, 0xdd0d7cc9L, 0x5005713cL, 0x270241aaL, 0xbe0b1010L,
		0xc90c2086L, 0x5768b525L, 0x206f85b3L, 0xb966d409L, 0xce61e49fL,
		0x5edef90eL, 0x29d9c998L, 0xb0d09822L, 0xc7d7a8b4L, 0x59b33d17L,
		0x2eb40d81L, 0xb7bd5c3bL, 0xc0ba6cadL, 0xedb88320L, 0x9abfb3b6L,
		0x03b6e20cL, 0x74b1d29aL, 0xead54739L, 0x9dd277afL, 0x04db2615L,
		0x73dc1683L, 0xe3630b12L, 0x94643b84L, 0x0d6d6a3eL, 0x7a6a5aa8L,
		0xe40ecf0bL, 0x9309ff9dL, 0x0a00ae27L, 0x7d079eb1L, 0xf00f9344L,
		0x8708a3d2L, 0x1e01f268L, 0x6906c2feL, 0xf762575dL, 0x806567cbL,
		0x196c3671L, 0x6e6b06e7L, 0xfed41b76L, 0x89d32be0L, 0x10da7a5aL,
		0x67dd4accL, 0xf9b9df6fL, 0x8ebeeff9L, 0x17b7be43L, 0x60b08ed5L,
		0xd6d6a3e8L, 0xa1d1937eL, 0x38d8c2c4L, 0x4fdff252L, 0xd1bb67f1L,
		0xa6bc5767L, 0x3fb506ddL, 0x48b2364bL, 0xd80d2bdaL, 0xaf0a1b4cL,
		0x36034af6L, 0x41047a60L, 0xdf60efc3L, 0xa867df55L, 0x316e8eefL,
		0x4669be79L, 0xcb61b38cL, 0xbc66831aL, 0x256fd2a0L, 0x5268e236L,
		0xcc0c7795L, 0xbb0b4703L, 0x220216b9L, 0x5505262fL, 0xc5ba3bbeL,
		0xb2bd0b28L, 0x2bb45a92L, 0x5cb36a04L, 0xc2d7ffa7L, 0xb5d0cf31L,
		0x2cd99e8bL, 0x5bdeae1dL, 0x9b64c2b0L, 0xec63f226L, 0x756aa39cL,
		0x026d930aL, 0x9c0906a9L, 0xeb0e363fL, 0x72076785L, 0x05005713L,
		0x95bf4a82L, 0xe2b87a14L, 0x7bb12baeL, 0x0cb61b38L, 0x92d28e9bL,
		0xe5d5be0dL, 0x7cdcefb7L, 0x0bdbdf21L, 0x86d3d2d4L, 0xf1d4e242L,
		0x68ddb3f8L, 0x1fda836eL, 0x81be16cdL, 0xf6b9265bL, 0x6fb077e1L,
		0x18b74777L, 0x88085ae6L, 0xff0f6a70L, 0x66063bcaL, 0x11010b5cL,
		0x8f659effL, 0xf862ae69L, 0x616bffd3L, 0x166ccf45L, 0xa00ae278L,
		0xd70dd2eeL, 0x4e048354L, 0x3903b3c2L, 0xa7672661L, 0xd06016f7L,
		0x4969474dL, 0x3e6e77dbL, 0xaed16a4aL, 0xd9d65adcL, 0x40df0b66L,
		0x37d83bf0L, 0xa9bcae53L, 0xdebb9ec5L, 0x47b2cf7fL, 0x30b5ffe9L,
		0xbdbdf21cL, 0xcabac28aL, 0x53b39330L, 0x24b4a3a6L, 0xbad03605L,
		0xcdd70693L, 0x54de5729L, 0x23d967bfL, 0xb3667a2eL, 0xc4614ab8L,
		0x5d681b02L, 0x2a6f2b94L, 0xb40bbe37L, 0xc30c8ea1L, 0x5a05df1bL,
		0x2d02ef8dL
	};

#define CRC32(c, b) (crc_table[((int)(c) ^ (b)) & 0xff] ^ ((c) >> 8))
#endif
#define DO1(buf)  crc = CRC32(crc, *buf++)
#define DO2(buf)  DO1(buf); DO1(buf)
#define DO4(buf)  DO2(buf); DO2(buf)
#define DO8(buf)  DO4(buf); DO4(buf)

	inline ulg crc32(ulg crc, const uch *buf, extent len) {
		if (buf == NULL) return 0L;
		crc = crc ^ 0xffffffffL;
		while (len >= 8) {
			DO8(buf); len -= 8;
		}
		if (len) do {
			DO1(buf);
		} while (--len);
		return crc ^ 0xffffffffL;  // (instead of ~c for 64-bit machines)
	}


	inline void update_keys(unsigned long *keys, char c) {
		keys[0] = CRC32(keys[0], c);
		keys[1] += keys[0] & 0xFF;
		keys[1] = keys[1] * 134775813L + 1;
		keys[2] = CRC32(keys[2], keys[1] >> 24);
	}
	inline char decrypt_byte(unsigned long *keys) {
		unsigned temp = ((unsigned)keys[2] & 0xffff) | 2;
		return (char)(((temp * (temp ^ 1)) >> 8) & 0xff);
	}
	inline char zencode(unsigned long *keys, char c) {
		int t = decrypt_byte(keys);
		update_keys(keys, c);
		return (char)(t^c);
	}







	inline int lustricmp(const TCHAR *sa, const TCHAR *sb) {
		for (const TCHAR *ca = sa, *cb = sb; ; ca++, cb++) {
			int ia = tolower(*ca), ib = tolower(*cb);
			if (ia == ib && ia == 0) return 0;
			if (ia == ib) continue;
			if (ia < ib) return -1;
			if (ia > ib) return 1;
		}
	}


	inline bool HasZipSuffix(const TCHAR *fn) {
		const TCHAR *ext = fn + _tcslen(fn);
		while (ext > fn && *ext != '.') ext--;
		if (ext == fn && *ext != '.') return false;
		if (lustricmp(ext, _T(".Z")) == 0) return true;
		if (lustricmp(ext, _T(".zip")) == 0) return true;
		if (lustricmp(ext, _T(".zoo")) == 0) return true;
		if (lustricmp(ext, _T(".arc")) == 0) return true;
		if (lustricmp(ext, _T(".lzh")) == 0) return true;
		if (lustricmp(ext, _T(".arj")) == 0) return true;
		if (lustricmp(ext, _T(".gz")) == 0) return true;
		if (lustricmp(ext, _T(".tgz")) == 0) return true;
		return false;
	}








	class TZip {
	public:
		TZip(const char *pwd) : hfout(0), mustclosehfout(false), hmapout(0), zfis(0), obuf(0), hfin(0), writ(0), oerr(false), hasputcen(false), ooffset(0), encwriting(false), encbuf(0), password(0), state(0) {
			if (pwd != 0 && *pwd != 0) {
				password = new char[strlen(pwd) + 1]; strcpy_s(password, strlen(pwd) + 1, pwd);
			}
		}
		~TZip() {
			if (state != 0) delete state; state = 0; if (encbuf != 0) delete[] encbuf; encbuf = 0; if (password != 0) delete[] password; password = 0;
		}

		// These variables say about the file we're writing into
		// We can write to pipe, file-by-handle, file-by-name, memory-to-memmapfile
		char *password;           // keep a copy of the password
		HANDLE hfout;             // if valid, we'll write here (for files or pipes)
		bool mustclosehfout;      // if true, we are responsible for closing hfout
		HANDLE hmapout;           // otherwise, we'll write here (for memmap)
		unsigned ooffset;         // for hfout, this is where the pointer was initially
		ZRESULT oerr;             // did a write operation give rise to an error?
		unsigned writ;            // how far have we written. This is maintained by Add, not write(), to avoid confusion over seeks
		bool ocanseek;            // can we seek?
		char *obuf;               // this is where we've locked mmap to view.
		unsigned int opos;        // current pos in the mmap
		unsigned int mapsize;     // the size of the map we created
		bool hasputcen;           // have we yet placed the central directory?
		bool encwriting;          // if true, then we'll encrypt stuff using 'keys' before we write it to disk
		unsigned long keys[3];    // keys are initialised inside Add()
		char *encbuf;             // if encrypting, then this is a temporary workspace for encrypting the data
		unsigned int encbufsize;  // (to be used and resized inside write(), and deleted in the destructor)
								  //
		TZipFileInfo *zfis;       // each file gets added onto this list, for writing the table at the end
		TState *state;            // we use just one state object per zip, because it's big (500k)

		inline ZRESULT Create(void *z, unsigned int len, DWORD flags);
		inline static unsigned sflush(void *param, const char *buf, unsigned *size);
		inline static unsigned swrite(void *param, const char *buf, unsigned size);
		inline unsigned int write(const char *buf, unsigned int size);
		inline bool oseek(unsigned int pos);
		inline ZRESULT GetMemory(void **pbuf, unsigned long *plen);
		inline ZRESULT Close();

		// some variables to do with the file currently being read:
		// I haven't done it object-orientedly here, just put them all
		// together, since OO didn't seem to make the design any clearer.
		ulg attr; iztimes times; ulg timestamp;  // all open_* methods set these
		bool iseekable; long isize, ired;         // size is not set until close() on pips
		ulg crc;                                 // crc is not set until close(). iwrit is cumulative
		HANDLE hfin; bool selfclosehf;           // for input files and pipes
		const char *bufin; unsigned int lenin, posin; // for memory
													  // and a variable for what we've done with the input: (i.e. compressed it!)
		ulg csize;                               // compressed size, set by the compression routines
												 // and this is used by some of the compression routines
		char buf[16384];


		inline 	ZRESULT open_file(const TCHAR *fn);
		inline ZRESULT open_handle(HANDLE hf, unsigned int len);
		inline ZRESULT open_mem(void *src, unsigned int len);
		inline ZRESULT open_dir();
		inline static unsigned sread(TState &s, char *buf, unsigned size);
		inline unsigned read(char *buf, unsigned size);
		inline ZRESULT iclose();

		inline ZRESULT ideflate(TZipFileInfo *zfi);
		inline ZRESULT istore();

		inline ZRESULT Add(const TCHAR *odstzn, void *src, unsigned int len, DWORD flags);
		inline ZRESULT AddCentral();

	};



	inline ZRESULT TZip::Create(void *z, unsigned int len, DWORD flags) {
		if (hfout != 0 || hmapout != 0 || obuf != 0 || writ != 0 || oerr != ZR_OK || hasputcen) return ZR_NOTINITED;
		//
		if (flags == ZIP_HANDLE) {
			HANDLE hf = (HANDLE)z;
			hfout = hf; mustclosehfout = false;
#ifdef DuplicateHandle
			BOOL res = DuplicateHandle(GetCurrentProcess(), hf, GetCurrentProcess(), &hfout, 0, FALSE, DUPLICATE_SAME_ACCESS);
			if (res) mustclosehandle = true;
#endif
			// now we have hfout. Either we duplicated the handle and we close it ourselves
			// (while the caller closes h themselves), or we couldn't duplicate it.
			DWORD res = GetFilePosZ(hfout);
			ocanseek = (res != 0xFFFFFFFF);
			ooffset = ocanseek ? res : 0;
			return ZR_OK;
		} else if (flags == ZIP_FILENAME) {
			const TCHAR *fn = (const TCHAR*)z;

			hfout = CreateFile(fn, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
			if (hfout == INVALID_HANDLE_VALUE) {
				hfout = 0; return ZR_NOFILE;
			}

			ocanseek = true;
			ooffset = 0;
			mustclosehfout = true;
			return ZR_OK;
		} else if (flags == ZIP_MEMORY) {
			unsigned int size = len;
			if (size == 0) return ZR_MEMSIZE;

			if (z != 0) obuf = (char*)z;
			else {
				hmapout = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, NULL);
				if (hmapout == NULL) return ZR_NOALLOC;
				obuf = (char*)MapViewOfFile(hmapout, FILE_MAP_ALL_ACCESS, 0, 0, size);
				if (obuf == 0) {
					CloseHandle(hmapout); hmapout = 0; return ZR_NOALLOC;
				}
			}

			ocanseek = true;
			opos = 0; mapsize = size;
			return ZR_OK;
		} else return ZR_ARGS;
	}

	inline unsigned TZip::sflush(void *param, const char *buf, unsigned *size) { // static
		if (*size == 0) return 0;
		TZip *zip = (TZip*)param;
		unsigned int writ = zip->write(buf, *size);
		if (writ != 0) *size = 0;
		return writ;
	}
	inline unsigned TZip::swrite(void *param, const char *buf, unsigned size) { // static
		if (size == 0) return 0;
		TZip *zip = (TZip*)param; return zip->write(buf, size);
	}
	inline unsigned int TZip::write(const char *buf, unsigned int size) {
		const char *srcbuf = buf;
		if (encwriting) {
			if (encbuf != 0 && encbufsize < size) {
				delete[] encbuf; encbuf = 0;
			}
			if (encbuf == 0) {
				encbuf = new char[size * 2]; encbufsize = size;
			}
			memcpy(encbuf, buf, size);
			for (unsigned int i = 0; i < size; i++) encbuf[i] = zencode(keys, encbuf[i]);
			srcbuf = encbuf;
		}
		if (obuf != 0) {
			if (opos + size >= mapsize) {
				oerr = ZR_MEMSIZE; return 0;
			}
			memcpy(obuf + opos, srcbuf, size);
			opos += size;
			return size;
		} else if (hfout != 0) {

			DWORD writ; WriteFile(hfout, srcbuf, size, &writ, NULL);
			return writ;
		}
		oerr = ZR_NOTINITED; return 0;
	}

	inline bool TZip::oseek(unsigned int pos) {
		if (!ocanseek) {
			oerr = ZR_SEEK; return false;
		}
		if (obuf != 0) {
			if (pos >= mapsize) {
				oerr = ZR_MEMSIZE; return false;
			}
			opos = pos;
			return true;
		} else if (hfout != 0) {

			SetFilePointer(hfout, pos + ooffset, NULL, FILE_BEGIN);
			return true;
		}
		oerr = ZR_NOTINITED; return 0;
	}

	inline ZRESULT TZip::GetMemory(void **pbuf, unsigned long *plen) { // When the user calls GetMemory, they're presumably at the end
																	   // of all their adding. In any case, we have to add the central
																	   // directory now, otherwise the memory we tell them won't be complete.
		if (!hasputcen) AddCentral(); hasputcen = true;
		if (pbuf != NULL) *pbuf = (void*)obuf;
		if (plen != NULL) *plen = writ;
		if (obuf == NULL) return ZR_NOTMMAP;
		return ZR_OK;
	}

	inline ZRESULT TZip::Close() { // if the directory hadn't already been added through a call to GetMemory,
								   // then we do it now
		ZRESULT res = ZR_OK; if (!hasputcen) res = AddCentral(); hasputcen = true;

		if (obuf != 0 && hmapout != 0) UnmapViewOfFile(obuf); obuf = 0;
		if (hmapout != 0) CloseHandle(hmapout); hmapout = 0;
		if (hfout != 0 && mustclosehfout) CloseHandle(hfout); hfout = 0; mustclosehfout = false;

		return res;
	}




	inline ZRESULT TZip::open_file(const TCHAR *fn) {
		hfin = 0; bufin = 0; selfclosehf = false; crc = CRCVAL_INITIAL; isize = 0; csize = 0; ired = 0;
		if (fn == 0) return ZR_ARGS;

		HANDLE hf = CreateFile(fn, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
		if (hf == INVALID_HANDLE_VALUE) return ZR_NOFILE;
		ZRESULT res = open_handle(hf, 0);
		if (res != ZR_OK) {
			CloseHandle(hf); return res;
		}

		selfclosehf = true;
		return ZR_OK;
	}
	inline ZRESULT TZip::open_handle(HANDLE hf, unsigned int len) {
		hfin = 0; bufin = 0; selfclosehf = false; crc = CRCVAL_INITIAL; isize = 0; csize = 0; ired = 0;
		if (hf == 0 || hf == INVALID_HANDLE_VALUE) return ZR_ARGS;
		bool canseek;

		DWORD res = SetFilePointer(hfout, 0, 0, FILE_CURRENT);
		canseek = (res != 0xFFFFFFFF);

		if (canseek) {
			ZRESULT res = GetFileInfo(hf, &attr, &isize, &times, &timestamp);
			if (res != ZR_OK) return res;

			SetFilePointer(hf, 0, NULL, FILE_BEGIN); // because GetFileInfo will have screwed it up
			iseekable = true; hfin = hf;
			return ZR_OK;
		} else {
			attr = 0x80000000;      // just a normal file
			isize = -1;            // can't know size until at the end
			if (len != 0) isize = len; // unless we were told explicitly!
			iseekable = false;
			WORD dosdate, dostime; GetNow(&times.atime, &dosdate, &dostime);
			times.mtime = times.atime;
			times.ctime = times.atime;
			timestamp = (WORD)dostime | (((DWORD)dosdate) << 16);
			hfin = hf;
			return ZR_OK;
		}
	}
	inline ZRESULT TZip::open_mem(void *src, unsigned int len) {
		hfin = 0; bufin = (const char*)src; selfclosehf = false; crc = CRCVAL_INITIAL; ired = 0; csize = 0; ired = 0;
		lenin = len; posin = 0;
		if (src == 0 || len == 0) return ZR_ARGS;
		attr = 0x80000000; // just a normal file
		isize = len;
		iseekable = true;
		WORD dosdate, dostime; GetNow(&times.atime, &dosdate, &dostime);
		times.mtime = times.atime;
		times.ctime = times.atime;
		timestamp = (WORD)dostime | (((DWORD)dosdate) << 16);
		return ZR_OK;
	}
	inline ZRESULT TZip::open_dir() {
		hfin = 0; bufin = 0; selfclosehf = false; crc = CRCVAL_INITIAL; isize = 0; csize = 0; ired = 0;
		attr = 0x41C00010; // a readable writable directory, and again directory
		isize = 0;
		iseekable = false;
		WORD dosdate, dostime; GetNow(&times.atime, &dosdate, &dostime);
		times.mtime = times.atime;
		times.ctime = times.atime;
		timestamp = (WORD)dostime | (((DWORD)dosdate) << 16);
		return ZR_OK;
	}

	inline unsigned TZip::sread(TState &s, char *buf, unsigned size) { // static
		TZip *zip = (TZip*)s.param;
		return zip->read(buf, size);
	}

	inline unsigned TZip::read(char *buf, unsigned size) {
		if (bufin != 0) {
			if (posin >= lenin) return 0; // end of input
			ulg red = lenin - posin;
			if (red > size) red = size;
			memcpy(buf, bufin + posin, red);
			posin += red;
			ired += red;
			crc = crc32(crc, (uch*)buf, red);
			return red;
		} else if (hfin != 0) {
			DWORD red;

			BOOL ok = ReadFile(hfin, buf, size, &red, NULL);
			if (!ok) return 0;

			ired += red;
			crc = crc32(crc, (uch*)buf, red);
			return red;
		} else {
			oerr = ZR_NOTINITED; return 0;
		}
	}

	inline ZRESULT TZip::iclose() {

		if (selfclosehf && hfin != 0) CloseHandle(hfin); hfin = 0;

		bool mismatch = (isize != -1 && isize != ired);
		isize = ired; // and crc has been being updated anyway
		if (mismatch) return ZR_MISSIZE;
		else return ZR_OK;
	}



	inline ZRESULT TZip::ideflate(TZipFileInfo *zfi) {
		if (state == 0) state = new TState();
		// It's a very big object! 500k! We allocate it on the heap, because PocketPC's
		// stack breaks if we try to put it all on the stack. It will be deleted lazily
		state->err = 0;
		state->readfunc = sread; state->flush_outbuf = sflush;
		state->param = this; state->level = 8; state->seekable = iseekable; state->err = NULL;
		// the following line will make ct_init realise it has to perform the init
		state->ts.static_dtree[0].dl.len = 0;
		// Thanks to Alvin77 for this crucial fix:
		state->ds.window_size = 0;
		//  I think that covers everything that needs to be initted.
		//
		bi_init(*state, buf, sizeof(buf), 1); // it used to be just 1024-size, not 16384 as here
		ct_init(*state, &zfi->att);
		lm_init(*state, state->level, &zfi->flg);
		ulg sz = deflate(*state);
		csize = sz;
		ZRESULT r = ZR_OK; if (state->err != NULL) r = ZR_FLATE;
		return r;
	}

	inline ZRESULT TZip::istore() {
		ulg size = 0;
		for (;;) {
			unsigned int cin = read(buf, 16384); if (cin <= 0 || cin == (unsigned int)EOF) break;
			unsigned int cout = write(buf, cin); if (cout != cin) return ZR_MISSIZE;
			size += cin;
		}
		csize = size;
		return ZR_OK;
	}





	__declspec(selectany) bool has_seeded = false;
	inline ZRESULT TZip::Add(const TCHAR *odstzn, void *src, unsigned int len, DWORD flags) {
		if (oerr) return ZR_FAILED;
		if (hasputcen) return ZR_ENDED;

		// if we use password encryption, then every isize and csize is 12 bytes bigger
		int passex = 0; if (password != 0 && flags != ZIP_FOLDER) passex = 12;

		// zip has its own notion of what its names should look like: i.e. dir/file.stuff
		TCHAR dstzn[MAX_PATH]; _tcsncpy_s(dstzn, MAX_PATH, odstzn, MAX_PATH); dstzn[MAX_PATH - 1] = 0;
		if (*dstzn == 0) return ZR_ARGS;
		TCHAR *d = dstzn; while (*d != 0) {
			if (*d == '\\') *d = '/'; d++;
		}
		bool isdir = (flags == ZIP_FOLDER);
		bool needs_trailing_slash = (isdir && dstzn[_tcslen(dstzn) - 1] != '/');
		int method = DEFLATE; if (isdir || HasZipSuffix(dstzn)) method = STORE;

		// now open whatever was our input source:
		ZRESULT openres;
		if (flags == ZIP_FILENAME) openres = open_file((const TCHAR*)src);
		else if (flags == ZIP_HANDLE) openres = open_handle((HANDLE)src, len);
		else if (flags == ZIP_MEMORY) openres = open_mem(src, len);
		else if (flags == ZIP_FOLDER) openres = open_dir();
		else return ZR_ARGS;
		if (openres != ZR_OK) return openres;

		// A zip "entry" consists of a local header (which includes the file name),
		// then the compressed data, and possibly an extended local header.

		// Initialize the local header
		TZipFileInfo zfi; zfi.nxt = NULL;
		strcpy_s(zfi.name, 260, "");
#ifdef UNICODE
		WideCharToMultiByte(CP_UTF8, 0, dstzn, -1, zfi.iname, MAX_PATH, 0, 0);
#else
		strncpy(zfi.iname, dstzn, MAX_PATH); zfi.iname[MAX_PATH - 1] = 0;
#endif
		zfi.nam = strlen(zfi.iname);
		if (needs_trailing_slash) {
			strcat_s(zfi.iname, "/"); zfi.nam++;
		}
		strcpy_s(zfi.zname, 260, "");
		zfi.extra = NULL; zfi.ext = 0;   // extra header to go after this compressed data, and its length
		zfi.cextra = NULL; zfi.cext = 0; // extra header to go in the central end-of-zip directory, and its length
		zfi.comment = NULL; zfi.com = 0; // comment, and its length
		zfi.mark = 1;
		zfi.dosflag = 0;
		zfi.att = (ush)BINARY;
		zfi.vem = (ush)0xB17; // 0xB00 is win32 os-code. 0x17 is 23 in decimal: zip 2.3
		zfi.ver = (ush)20;    // Needs PKUNZIP 2.0 to unzip it
		zfi.tim = timestamp;
		// Even though we write the header now, it will have to be rewritten, since we don't know compressed size or crc.
		zfi.crc = 0;            // to be updated later
		zfi.flg = 8;            // 8 means 'there is an extra header'. Assume for the moment that we need it.
		if (password != 0 && !isdir) zfi.flg = 9;  // and 1 means 'password-encrypted'
		zfi.lflg = zfi.flg;     // to be updated later
		zfi.how = (ush)method;  // to be updated later
		zfi.siz = (ulg)(method == STORE && isize >= 0 ? isize + passex : 0); // to be updated later
		zfi.len = (ulg)(isize);  // to be updated later
		zfi.dsk = 0;
		zfi.atx = attr;
		zfi.off = writ + ooffset;         // offset within file of the start of this local record
										  // stuff the 'times' structure into zfi.extra

										  // nb. apparently there's a problem with PocketPC CE(zip)->CE(unzip) fails. And removing the following block fixes it up.
		char xloc[EB_L_UT_SIZE]; zfi.extra = xloc;  zfi.ext = EB_L_UT_SIZE;
		char xcen[EB_C_UT_SIZE]; zfi.cextra = xcen; zfi.cext = EB_C_UT_SIZE;
		xloc[0] = 'U';
		xloc[1] = 'T';
		xloc[2] = EB_UT_LEN(3);       // length of data part of e.f.
		xloc[3] = 0;
		xloc[4] = EB_UT_FL_MTIME | EB_UT_FL_ATIME | EB_UT_FL_CTIME;
		xloc[5] = (char)(times.mtime);
		xloc[6] = (char)(times.mtime >> 8);
		xloc[7] = (char)(times.mtime >> 16);
		xloc[8] = (char)(times.mtime >> 24);
		xloc[9] = (char)(times.atime);
		xloc[10] = (char)(times.atime >> 8);
		xloc[11] = (char)(times.atime >> 16);
		xloc[12] = (char)(times.atime >> 24);
		xloc[13] = (char)(times.ctime);
		xloc[14] = (char)(times.ctime >> 8);
		xloc[15] = (char)(times.ctime >> 16);
		xloc[16] = (char)(times.ctime >> 24);
		memcpy(zfi.cextra, zfi.extra, EB_C_UT_SIZE);
		zfi.cextra[EB_LEN] = EB_UT_LEN(1);


		// (1) Start by writing the local header:
		int r = putlocal(&zfi, swrite, this);
		if (r != ZE_OK) {
			iclose(); return ZR_WRITE;
		}
		writ += 4 + LOCHEAD + (unsigned int)zfi.nam + (unsigned int)zfi.ext;
		if (oerr != ZR_OK) {
			iclose(); return oerr;
		}

		// (1.5) if necessary, write the encryption header
		keys[0] = 305419896L;
		keys[1] = 591751049L;
		keys[2] = 878082192L;
		for (const char *cp = password; cp != 0 && *cp != 0; cp++) update_keys(keys, *cp);
		// generate some random bytes

		DWORD a = (size_t)GetDesktopWindow()&INT_MAX;
		if (!has_seeded) srand(GetTickCount() ^ a);

		char encbuf[12]; for (int i = 0; i < 12; i++) encbuf[i] = (char)((rand() >> 7) & 0xff);
		encbuf[11] = (char)((zfi.tim >> 8) & 0xff);
		for (int ei = 0; ei < 12; ei++) encbuf[ei] = zencode(keys, encbuf[ei]);
		if (password != 0 && !isdir) {
			swrite(this, encbuf, 12); writ += 12;
		}

		//(2) Write deflated/stored file to zip file
		ZRESULT writeres = ZR_OK;
		encwriting = (password != 0 && !isdir);  // an object member variable to say whether we write to disk encrypted
		if (!isdir && method == DEFLATE) writeres = ideflate(&zfi);
		else if (!isdir && method == STORE) writeres = istore();
		else if (isdir) csize = 0;
		encwriting = false;
		iclose();
		writ += csize;
		if (oerr != ZR_OK) return oerr;
		if (writeres != ZR_OK) return ZR_WRITE;

		// (3) Either rewrite the local header with correct information...
		bool first_header_has_size_right = (zfi.siz == csize + passex);
		zfi.crc = crc;
		zfi.siz = csize + passex;
		zfi.len = isize;
		if (ocanseek && (password == 0 || isdir)) {
			zfi.how = (ush)method;
			if ((zfi.flg & 1) == 0) zfi.flg &= ~8; // clear the extended local header flag
			zfi.lflg = zfi.flg;
			// rewrite the local header:
			if (!oseek(zfi.off - ooffset)) return ZR_SEEK;
			if ((r = putlocal(&zfi, swrite, this)) != ZE_OK) return ZR_WRITE;
			if (!oseek(writ)) return ZR_SEEK;
		} else { // (4) ... or put an updated header at the end
			if (zfi.how != (ush)method) return ZR_NOCHANGE;
			if (method == STORE && !first_header_has_size_right) return ZR_NOCHANGE;
			if ((r = putextended(&zfi, swrite, this)) != ZE_OK) return ZR_WRITE;
			writ += 16L;
			zfi.flg = zfi.lflg; // if flg modified by inflate, for the central index
		}
		if (oerr != ZR_OK) return oerr;

		// Keep a copy of the zipfileinfo, for our end-of-zip directory
		char *cextra = new char[zfi.cext]; memcpy(cextra, zfi.cextra, zfi.cext); zfi.cextra = cextra;
		TZipFileInfo *pzfi = new TZipFileInfo; memcpy(pzfi, &zfi, sizeof(zfi));
		if (zfis == NULL) zfis = pzfi;
		else {
			TZipFileInfo *z = zfis; while (z->nxt != NULL) z = z->nxt; z->nxt = pzfi;
		}
		return ZR_OK;
	}

	inline ZRESULT TZip::AddCentral() { // write central directory
		int numentries = 0;
		ulg pos_at_start_of_central = writ;
		//ulg tot_unc_size=0, tot_compressed_size=0;
		bool okay = true;
		for (TZipFileInfo *zfi = zfis; zfi != NULL; ) {
			if (okay) {
				int res = putcentral(zfi, swrite, this);
				if (res != ZE_OK) okay = false;
			}
			writ += 4 + CENHEAD + (unsigned int)zfi->nam + (unsigned int)zfi->cext + (unsigned int)zfi->com;
			//tot_unc_size += zfi->len;
			//tot_compressed_size += zfi->siz;
			numentries++;
			//
			TZipFileInfo *zfinext = zfi->nxt;
			if (zfi->cextra != 0) delete[] zfi->cextra;
			delete zfi;
			zfi = zfinext;
		}
		ulg center_size = writ - pos_at_start_of_central;
		if (okay) {
			int res = putend(numentries, center_size, pos_at_start_of_central + ooffset, 0, NULL, swrite, this);
			if (res != ZE_OK) okay = false;
			writ += 4 + ENDHEAD + 0;
		}
		if (!okay) return ZR_WRITE;
		return ZR_OK;
	}





	__declspec(selectany)ZRESULT lasterrorZ = ZR_OK;

	inline unsigned int FormatZipMessageZ(ZRESULT code, char *buf, unsigned int len) {
		if (code == ZR_RECENT) code = lasterrorZ;
		const char *msg = "unknown zip result code";
		switch (code) {
			case ZR_OK: msg = "Success"; break;
			case ZR_NODUPH: msg = "Culdn't duplicate handle"; break;
			case ZR_NOFILE: msg = "Couldn't create/open file"; break;
			case ZR_NOALLOC: msg = "Failed to allocate memory"; break;
			case ZR_WRITE: msg = "Error writing to file"; break;
			case ZR_NOTFOUND: msg = "File not found in the zipfile"; break;
			case ZR_MORE: msg = "Still more data to unzip"; break;
			case ZR_CORRUPT: msg = "Zipfile is corrupt or not a zipfile"; break;
			case ZR_READ: msg = "Error reading file"; break;
			case ZR_ARGS: msg = "Caller: faulty arguments"; break;
			case ZR_PARTIALUNZ: msg = "Caller: the file had already been partially unzipped"; break;
			case ZR_NOTMMAP: msg = "Caller: can only get memory of a memory zipfile"; break;
			case ZR_MEMSIZE: msg = "Caller: not enough space allocated for memory zipfile"; break;
			case ZR_FAILED: msg = "Caller: there was a previous error"; break;
			case ZR_ENDED: msg = "Caller: additions to the zip have already been ended"; break;
			case ZR_ZMODE: msg = "Caller: mixing creation and opening of zip"; break;
			case ZR_NOTINITED: msg = "Zip-bug: internal initialisation not completed"; break;
			case ZR_SEEK: msg = "Zip-bug: trying to seek the unseekable"; break;
			case ZR_MISSIZE: msg = "Zip-bug: the anticipated size turned out wrong"; break;
			case ZR_NOCHANGE: msg = "Zip-bug: tried to change mind, but not allowed"; break;
			case ZR_FLATE: msg = "Zip-bug: an internal error during flation"; break;
		}
		unsigned int mlen = (unsigned int)strlen(msg);
		if (buf == 0 || len == 0) return mlen;
		unsigned int n = mlen; if (n + 1 > len) n = len - 1;
		//strncpy(buf,msg,n);
		memcpy(buf, msg, n);
		buf[n] = 0;
		return mlen;
	}



	typedef struct {
		DWORD flag;
		TZip *zip;
	} TZipHandleData;


	inline HZIP CreateZipInternal(void *z, unsigned int len, DWORD flags, const char *password) {
		TZip *zip = new TZip(password);
		lasterrorZ = zip->Create(z, len, flags);
		if (lasterrorZ != ZR_OK) {
			delete zip; return 0;
		}
		TZipHandleData *han = new TZipHandleData;
		han->flag = 2; han->zip = zip; return (HZIP)han;
	}
	inline HZIP CreateZipHandle(HANDLE h, const char *password) {
		return CreateZipInternal(h, 0, ZIP_HANDLE, password);
	}
	inline HZIP CreateZip(const TCHAR *fn, const char *password) {
		return CreateZipInternal((void*)fn, 0, ZIP_FILENAME, password);
	}
	inline HZIP CreateZip(void *z, unsigned int len, const char *password) {
		return CreateZipInternal(z, len, ZIP_MEMORY, password);
	}


	inline ZRESULT ZipAddInternal(HZIP hz, const TCHAR *dstzn, void *src, unsigned int len, DWORD flags) {
		if (hz == 0) {
			lasterrorZ = ZR_ARGS; return ZR_ARGS;
		}
		TZipHandleData *han = (TZipHandleData*)hz;
		if (han->flag != 2) {
			lasterrorZ = ZR_ZMODE; return ZR_ZMODE;
		}
		TZip *zip = han->zip;
		lasterrorZ = zip->Add(dstzn, src, len, flags);
		return lasterrorZ;
	}
	inline ZRESULT ZipAdd(HZIP hz, const TCHAR *dstzn, const TCHAR *fn) {
		return ZipAddInternal(hz, dstzn, (void*)fn, 0, ZIP_FILENAME);
	}
	inline ZRESULT ZipAdd(HZIP hz, const TCHAR *dstzn, void *src, unsigned int len) {
		return ZipAddInternal(hz, dstzn, src, len, ZIP_MEMORY);
	}
	inline ZRESULT ZipAddHandle(HZIP hz, const TCHAR *dstzn, HANDLE h) {
		return ZipAddInternal(hz, dstzn, h, 0, ZIP_HANDLE);
	}
	inline ZRESULT ZipAddHandle(HZIP hz, const TCHAR *dstzn, HANDLE h, unsigned int len) {
		return ZipAddInternal(hz, dstzn, h, len, ZIP_HANDLE);
	}
	inline ZRESULT ZipAddFolder(HZIP hz, const TCHAR *dstzn) {
		return ZipAddInternal(hz, dstzn, 0, 0, ZIP_FOLDER);
	}



	inline ZRESULT ZipGetMemory(HZIP hz, void **buf, unsigned long *len) {
		if (hz == 0) {
			if (buf != 0) *buf = 0; if (len != 0) *len = 0; lasterrorZ = ZR_ARGS; return ZR_ARGS;
		}
		TZipHandleData *han = (TZipHandleData*)hz;
		if (han->flag != 2) {
			lasterrorZ = ZR_ZMODE; return ZR_ZMODE;
		}
		TZip *zip = han->zip;
		lasterrorZ = zip->GetMemory(buf, len);
		return lasterrorZ;
	}

	inline ZRESULT CloseZipZ(HZIP hz) {
		if (hz == 0) {
			lasterrorZ = ZR_ARGS; return ZR_ARGS;
		}
		TZipHandleData *han = (TZipHandleData*)hz;
		if (han->flag != 2) {
			lasterrorZ = ZR_ZMODE; return ZR_ZMODE;
		}
		TZip *zip = han->zip;
		lasterrorZ = zip->Close();
		delete zip;
		delete han;
		return lasterrorZ;
	}

	inline bool IsZipHandleZ(HZIP hz) {
		if (hz == 0) return false;
		TZipHandleData *han = (TZipHandleData*)hz;
		return (han->flag == 2);
	}
}
#endif //_WIN32
#endif //DOXYGEN
#endif