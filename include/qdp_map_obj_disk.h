// -*- C++ -*-
/*! \file
 *  \brief A Map Object that works lazily from Disk
 */


#ifndef __qdp_map_obj_disk_h__
#define __qdp_map_obj_disk_h__

#include "qdp_map_obj.h"
#include <string>
#include <unistd.h>

namespace QDP
{

  namespace MapObjDiskEnv { 
    typedef unsigned int file_version_t;
   
    const std::string& getFileMagic();
  };




  //----------------------------------------------------------------------------
  //! A wrapper over maps
  template<typename K, typename V>
  class MapObjectDisk : public MapObject<K,V>
  {
  public:
    typedef std::iostream::pos_type pos_type;  // position in buffer
    typedef std::iostream::off_type off_type;  // offset in buffer

  public:
    //! Empty constructor
    MapObjectDisk() : file_version(1), state(INIT), debug(false) {}

    //! Set debugging level
    void setDebug(int level);

    //! Get debugging level
    bool getDebug() const {return debug;}

    //! Open a file
    void open(const std::string& file, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out);

    //! Check if a DB file exists before opening.
    bool fileExists(const std::string& file) const {
      bool ret = 0;
      if (Layout::primaryNode()) 
	ret = ::access(file.c_str(), F_OK);

      Internal::broadcast(ret);
      return (ret == 0) ? true : false;
    }


    //! Finalizes object
    ~MapObjectDisk();

    /**
     * Insert a pair of data and key into the database
     * @param key a key
     * @param val a user provided data
     *
     * @return 0 on successful write, -1 on failure with proper errno set
     */
    int insert(const K& key, const V& val);

    /**
     * Get data for a given key
     * @param key user supplied key
     * @param data after the call data will be populated
     * @return 0 on success, otherwise the key not found
     */
    int get(const K& key, V& val);


    /**
     * Flush database in memory to disk
     */
    void flush();


    /**
     * Does this key exist in the store
     * @param key a key object
     * @return true if the answer is yes
     */
    bool exist(const K& key) const;

    /** 
     * The number of elements
     */
    unsigned int size() const {return static_cast<unsigned long>(src_map.size());}

    /**
     * Return all available keys to user
     * @param keys user suppled an empty vector which is populated
     * by keys after this call.
     */
    void keys(std::vector<K>& keys_) const;
    
    /**
     * Insert user data into the  metadata database
     *
     * @param user_data user supplied data
     * @return returns 0 if success, else failure
     */
    int insertUserdata(const std::string& user_data);
    
    /**
     * Get user user data from the metadata database
     *
     * @param user_data user supplied buffer to store user data
     * @return returns 0 if success. Otherwise failure.
     */
    int getUserdata(std::string& user_data);

  private:
    //! Type for the map
    typedef std::map<K, pos_type> MapType_t;

    //! State 
    enum State {INIT, UNCHANGED, MODIFIED};

    //! State of db
    State state;

    //! Debugging
    bool debug;

    //! File related stuff. Unsigned int is as close to uint32 as I can get
    MapObjDiskEnv::file_version_t file_version;
    
    //! Usual begin iterator
    typename MapType_t::const_iterator begin() const {return src_map.begin();}
    
    //! Usual end iterator
    typename MapType_t::const_iterator end() const {return src_map.end();}
    
    //! Map of objects
    mutable MapType_t src_map;
    
    //! The parameters
    std::string filename;
    
    //! Metadata
    std::string user_data;

    //! Reader and writer interfaces
    mutable BinaryFileReaderWriter streamer;
    
    //! Open a new DB, and will write map
    void openWrite(const std::string& file, std::ios_base::openmode mode);

    //! Open an existing DB, and read map
    void openRead(const std::string& file, std::ios_base::openmode mode);

    // Internal Utility: Create/Skip past header
    void writeSkipHeader(void);
    
    //! Internal Utility: Read/Check header 
    pos_type readCheckHeader(void);
    
    //! Internal Utility: Dump the map to disk
    void writeMapBinary(void);  
    
    //! Internal Utility: Read the map from disk
    void readMapBinary(const pos_type& md_start);
    
    //! Internal Utility: Close File after write mode
    void closeWrite(void);
    
    //! Sink State for errors:
    void errorState(const std::string err) const {
      throw err;
    }
  };


  
  /* ****************** IMPLEMENTATIONS ********************** */
  
  //! Set debugging level
  template<typename K, typename V>
  void 
  MapObjectDisk<K,V>::setDebug(int level)
  {
    if (level > 0)
      debug = true;
  }


  //! Open a file
  template<typename K, typename V>
  void 
  MapObjectDisk<K,V>::open(const std::string& file, std::ios_base::openmode mode)
  {
    if (fileExists(filename) || ((mode & std::ios_base::trunc) == 0))
    {
      openRead(file, mode);
    }
    else
    {
      openWrite(file, mode);
    }
  }


  //! Open a new DB, and will write map
  template<typename K, typename V>
  void
  MapObjectDisk<K,V>::openWrite(const std::string& file, std::ios_base::openmode mode)
  {
    switch(state) { 
    case INIT: 
    {
      filename = file;

      QDPIO::cout << "MapObjectDisk: opening file " << filename
		  << " for writing" << endl;
      
      streamer.open(filename, mode);
          
      if (debug) {
	QDPIO::cout << "Writing file magic: len= " << MapObjDiskEnv::getFileMagic().length() << endl;
      }

      // Write string
      streamer.writeDesc(MapObjDiskEnv::getFileMagic());
      
      if (debug) {
	QDPIO::cout << "Wrote magic. Current Position: " << streamer.currentPosition() << endl;
      }
      
      write(streamer, (MapObjDiskEnv::file_version_t)file_version);
    
      if (debug) {
	QDPIO::cout << "Wrote Version. Current Position is: " << streamer.currentPosition() << endl;
      }
      
      if (debug) {
	QDPIO::cout << "Writing User Data string=" << user_data << endl;
      }
      writeDesc(streamer, user_data);
      
      if (debug) {
	QDPIO::cout << "Wrote User Data string. Current Position is: " << streamer.currentPosition() << endl;
      }
      
      pos_type dummypos = static_cast<pos_type>(streamer.currentPosition());
    
      if (debug) {
	int user_len = user_data.length();
	Internal::broadcast(user_len);

	QDPIO::cout << "Sanity Check 1" << endl; ;
	pos_type cur_pos = streamer.currentPosition();
	pos_type exp_pos = 
	  static_cast<pos_type>(MapObjDiskEnv::getFileMagic().length()+sizeof(int))
	  +static_cast<pos_type>(user_len+sizeof(int))
	  +static_cast<pos_type>(sizeof(MapObjDiskEnv::file_version_t));

	QDPIO::cout << "cur pos=" << cur_pos << " expected " << exp_pos << endl;

	if ( cur_pos != exp_pos ) {
	  QDPIO::cout << "ERROR: Sanity Check 1 failed." << endl;
	  QDPIO::cout << "cur pos=" << cur_pos << " expected " << exp_pos << endl;
	  QDP_abort(1);
	}
      }
    
      /* Write a dummy link - make room for it */
      streamer.writeArray((char *)&dummypos, sizeof(pos_type), 1);
      
      if (debug) {
	QDPIO::cout << "Wrote dummy link: Current Position " << streamer.currentPosition() << endl;
	int user_len = user_data.length();
	Internal::broadcast(user_len);

	QDPIO::cout << "Sanity Check 2" << endl;
	pos_type cur_pos = streamer.currentPosition();
	pos_type exp_pos = 
	  static_cast<pos_type>(MapObjDiskEnv::getFileMagic().length()+sizeof(int))
	  + static_cast<pos_type>(user_len+sizeof(int))
	  + static_cast<pos_type>(sizeof(MapObjDiskEnv::file_version_t))
	  + static_cast<pos_type>(sizeof(pos_type));

	if ( cur_pos != exp_pos ) {
	  QDPIO::cout << "Cur pos = " << cur_pos << endl;
	  QDPIO::cout << "Expected: " << exp_pos << endl;
	  QDPIO::cout << "ERROR: Sanity Check 2 failed." << endl;
	  QDP_abort(1);
	}
      }
      
      // Advance state machine state
      state = MODIFIED;
      break;      
    }

    default:
      errorState("MapOjectDisk: openWrite called from invalid state");
      break;
    }
  
    return;
  }

 

  //! Open an existing DB, and read map
  template<typename K, typename V>
  void
  MapObjectDisk<K,V>::openRead(const std::string& file, std::ios_base::openmode mode)
  {  
    switch (state) { 
    case INIT:
    {
      filename = file;

      QDPIO::cout << "MapObjectDisk: opening file " << filename
		  << " for reading" << endl;
      
      // Open the reader
      streamer.open(filename, mode);
	
      QDPIO::cout << "MapObjectDisk: reading and checking header" << endl;

      pos_type md_start = readCheckHeader();
	
      // Seek to metadata
      QDPIO::cout << "MapObjectDisk: reading key/fileposition data" << endl;
	
      /* Read the map in (metadata) */
      readMapBinary(md_start);
	
      /* And we are done */
      state = UNCHANGED;
    }
    break;
    default:
      errorState("MapObjectDisk: openRead() called from invalid state");
      break;
    }
    
    return;
  }


  
  //! Destructor
  template<typename K, typename V>
  MapObjectDisk<K,V>::~MapObjectDisk() 
  {
    switch(state) { 
    case UNCHANGED:
      if( streamer.is_open() ) { 
	streamer.close();
      }
      break;
    case MODIFIED:
      closeWrite(); // This finalizes files for us
      if( streamer.is_open() ) { 
	streamer.close();
      }
      break;
    case INIT:
      break;
    default:
      errorState("~MapObjectDisk: destructor called from invalid state");
      break;
    }
  }
  

  //! Destructor
  template<typename K, typename V>
  void
  MapObjectDisk<K,V>::flush() 
  {
    switch(state) { 
    case MODIFIED:
      closeWrite();  // not optimal
      state = UNCHANGED;
      break;
    case UNCHANGED:
      break;
    case INIT:
      break;
    default:
      break;
    }
  }
  

  //! Dump keys
  template<typename K, typename V>
  void
  MapObjectDisk<K,V>::keys(std::vector<K>& keys_) const 
  {
    if( streamer.is_open() ) 
    {
      typename MapType_t::const_iterator iter;
      for(iter  = src_map.begin();
	  iter != src_map.end();
	  ++iter) { 
	keys_.push_back(iter->first);
      }
    }
  }
    

  /**
   * Insert user data into the  metadata database
   */
  template<typename K, typename V>
  int 
  MapObjectDisk<K,V>::insertUserdata(const std::string& _user_data)
  {
    int ret = 0;
    switch(state) { 
    case INIT:
      user_data = _user_data;
      break;

    case UNCHANGED:
    case MODIFIED:
      ret = 1;
      break;

    default:
      errorState("MapObjectDisk::insertUserdata() called from invalid state");
      break;
    }

    return ret;
  }

    
  /**
   * Get user user data from the metadata database
   *
   * @param user_data user supplied buffer to store user data
   * @return returns 0 if success. Otherwise failure.
   */
  template<typename K, typename V>
  int
  MapObjectDisk<K,V>::getUserdata(std::string& _user_data)
  {
    int ret = 0;

    switch(state) { 
    case INIT:
    {
      ret = 1;
      break;
    }
    case UNCHANGED:
    case MODIFIED:
    {
      user_data = _user_data;
      break;
    }
    default:
      errorState("MapObjectDisk::getUserdata called from unknown state");
      break;
    }

    return ret;
  }


  /*! 
   * Insert a value into the Map.
   */
  template<typename K, typename V>
  int 
  MapObjectDisk<K,V>::insert(const K& key, const V& val) 
  {
    int ret = 0;

    switch (state)  { 
    case MODIFIED :
    case UNCHANGED : {
      //  Find key
      if (exist(key)) { 
	// Key does exist
	pos_type wpos = static_cast<pos_type>(src_map[key]);
	if (debug) {
	  QDPIO::cout << "Found key to update. Position is " << wpos << endl;
	}
	
	streamer.seek(wpos);
	
	if (debug) {
	  QDPIO::cout << "Sought write position. Current Position: " << streamer.currentPosition() << endl;
	}
	streamer.resetChecksum(); // Reset checksum. It gets calculated on write.
	write(streamer, val);
	
	if (debug) {	
	  QDPIO::cout << "Wrote value to disk. Current Position: " << streamer.currentPosition() << endl;
	}
	write(streamer, streamer.getChecksum()); // Write Checksum
	streamer.flush(); // Sync the file
	
	if (debug) {
	  QDPIO::cout << "Wrote checksum " << streamer.getChecksum() << " to disk. Current Position: " << streamer.currentPosition() << endl;
	}

	// Done
	state = MODIFIED;
      }
      else {
	// Key does not exist

	// Make note of current writer position
	pos_type pos=streamer.currentPosition();
      
	// Insert pos into map
	src_map.insert(std::make_pair(key,pos));
     
	streamer.resetChecksum();

	// Add position to the map
	StopWatch swatch;
	swatch.reset();
	swatch.start();

	write(streamer, val); // DO write
	swatch.stop();
     
	// Get diagnostics.
	pos_type end_pos = static_cast<pos_type>(streamer.currentPosition());
	double MiBWritten = (double)(end_pos - pos)/(double)(1024*1024);
	double time = swatch.getTimeInSeconds();

	QDPIO::cout << " wrote: " << MiBWritten << " MiB. Time: " << time << " sec. Write Bandwidth: " << MiBWritten/time<<endl;

	if (debug) {
	  QDPIO::cout << "Wrote value to disk. Current Position: " << streamer.currentPosition() << endl;
	}

	write(streamer, streamer.getChecksum()); // Write Checksum
	streamer.flush();
	
	if (debug) {
	  QDPIO::cout << "Wrote checksum " << streamer.getChecksum() << " to disk. Current Position: " << streamer.currentPosition() << endl;
	}

	// Done
	state = MODIFIED;
      }
      break;
    }
    default:
      ret = 1;
      break;
    }

    return ret;
  }



  /*! 
   * Lookup an item in the map.
   */
  template<typename K, typename V>
  int 
  MapObjectDisk<K,V>::get(const K& key, V& val)
  { 
    int ret = 0;

    switch(state) { 
    case UNCHANGED: // Deliberate fallthrough
    case MODIFIED: {
      if ( exist(key) ) 
      {
	// If key exists find file offset
	pos_type pos = src_map.find(key)->second;

	// Do the seek and time it 
	StopWatch swatch;

	swatch.reset();
	swatch.start();
	streamer.seek(pos);
	swatch.stop();
	double seek_time = swatch.getTimeInSeconds();

	// Reset the checkums
	streamer.resetChecksum();

	// Grab start pos: We've just seeked it
	pos_type start_pos = pos;

	// Time the read
	swatch.reset();
	swatch.start();
	read(streamer, val);
	swatch.stop();

	double read_time = swatch.getTimeInSeconds();
	pos_type end_pos = streamer.currentPosition();

	// Print data
	double MiBRead = (double)(end_pos-start_pos)/(double)(1024*1024);
	QDPIO::cout << " seek time: " << seek_time 
		    << " sec. read time: " << read_time 
		    << "  " << MiBRead <<" MiB, " << MiBRead/read_time << " MiB/sec" << endl;


	if (debug) { 
	  QDPIO::cout << "Read record. Current position: " << streamer.currentPosition() << endl;
	}

	QDPUtil::n_uint32_t calc_checksum=streamer.getChecksum();
	QDPUtil::n_uint32_t read_checksum;
	read(streamer, read_checksum);

	if (debug) {
	  QDPIO::cout << " Record checksum: " << read_checksum << "  Current Position: " << streamer.currentPosition() << endl;
	}

	if( read_checksum != calc_checksum ) { 
	  QDPIO::cout << "Mismatched Checksums: Expected: " << calc_checksum << " but read " << read_checksum << endl;
	  QDP_abort(1);
	}

	if (debug) {
	  QDPIO::cout << "  Checksum OK!" << endl;
	}
      }
      else {
	ret = 1;
      }
      break;
    }
    default:
      ret = 1;
      break;
    }

    return ret;
  }
  
  
  /**
   * Does this key exist in the store
   * @param key a key object
   * @return true if the answer is yes
   */
  template<typename K, typename V>
  bool 
  MapObjectDisk<K,V>::exist(const K& key) const 
  {
    return (src_map.find(key) == src_map.end()) ? false : true;
  }
  
  

  /***************** UTILITY ******************/


  //! Skip past header
  template<typename K, typename V>
  void 
  MapObjectDisk<K,V>::writeSkipHeader(void) 
  { 
    switch(state) { 
    case MODIFIED: {
      if ( streamer.is_open() ) 
      { 
	int user_len = user_data.length();
	Internal::broadcast(user_len);
	
	streamer.seek( MapObjDiskEnv::getFileMagic().length() + sizeof(int)
		       + user_len + sizeof(int)
		       + sizeof(MapObjDiskEnv::file_version_t) );
      }
      else { 
	QDPIO::cerr << "Attempting writeSkipHeader, not in write mode" <<endl;
	QDP_abort(1);
      }
    }
    break;
    default:
      errorState("MapObjectDisk: writeSkipHeader() called not in MODIFIED state");
      break;
    }
  }
  
  //! Check the header 
  template<typename K, typename V>
  typename MapObjectDisk<K,V>::pos_type 
  MapObjectDisk<K,V>::readCheckHeader(void) 
  {
    pos_type md_position = 0;
    if( streamer.is_open() ) 
    {
      if (debug) {
	QDPIO::cout << "Rewinding File" << endl;
      }
      
      streamer.rewind();
      
      std::string read_magic;
      streamer.readDesc(read_magic);
      
      // Check magic
      if (read_magic != MapObjDiskEnv::getFileMagic()) { 
	QDPIO::cerr << "Magic String Wrong: Expected: " << MapObjDiskEnv::getFileMagic() 
		    << " but read: " << read_magic << endl;
	QDP_abort(1);
      }

      if (debug) {
	QDPIO::cout << "Read File Magic. Current Position: " << streamer.currentPosition() << endl;
      }
      
      MapObjDiskEnv::file_version_t read_version;
      read(streamer, read_version);
      
      if (debug) {
	QDPIO::cout << "Read File Verion. Current Position: " << streamer.currentPosition() << endl;
      }
      
      // Check version
      QDPIO::cout << "MapObjectDisk: file has version: " << read_version << endl;
      
      QDP::readDesc(streamer, user_data);
      if (debug) {
	QDPIO::cout << "User data. String=" << user_data << ". Current Position: " << streamer.currentPosition() << endl;
      }
      
      // Read MD location
      streamer.readArray((char *)&md_position, sizeof(pos_type), 1);

      if (debug) {
	QDPIO::cout << "Read MD Location. Current position: " << streamer.currentPosition() << endl;
      }
      
      if (debug) {
	QDPIO::cout << "Metadata starts at position: " << md_position << endl;
      }
	
    }
    else { 
      QDPIO::cerr << "readCheckHeader needs reader mode to be opened. It is not" << endl;
      QDP_abort(1);
    }

    return md_position;
  }

  //! Dump the map
  // Private utility function -- no one else should use.
  template<typename K, typename V>
  void
  MapObjectDisk<K,V>::writeMapBinary(void)  
  {
    unsigned int map_size = src_map.size();

    streamer.resetChecksum();
    write(streamer, map_size);
    if (debug) {
      QDPIO::cout << "Wrote map size: " << map_size << " entries.  Current position : " << streamer.currentPosition() << endl;
    }
    
    typename MapType_t::const_iterator iter;
    for(iter  = src_map.begin();
	iter != src_map.end();
	++iter) 
    { 
      K key = iter->first;
      pos_type pos=iter->second;
      
      write(streamer, key); 
      streamer.writeArray((char *)&pos,sizeof(pos_type),1);
      
      if (debug) {
	QDPIO::cout << "Wrote Key/Position pair:  Current Position: " << streamer.currentPosition() << endl;
      }
    }
    write(streamer, streamer.getChecksum());
    QDPIO::cout << "Wrote Checksum On Map: " << streamer.getChecksum() << endl;
    streamer.flush();
  }
  
  //! read the map 
  // assume positioned at start of map data
  // Private utility function -- no one else should use.
  template<typename K, typename V>
  void 
  MapObjectDisk<K,V>::readMapBinary(const pos_type& md_start)
  {
    streamer.seek(md_start);
    streamer.resetChecksum();

    if (debug) {
      QDPIO::cout << "Sought start of metadata. Current position: " << streamer.currentPosition() << endl;
    }
    
    unsigned int num_records;
    read(streamer, num_records);

    if (debug) {
      QDPIO::cout << "Read num of entries: " << num_records << " records. Current Position: " << streamer.currentPosition() << endl;
    }
    
    for(unsigned int i=0; i < num_records; i++) { 
      pos_type rpos;
      K key;
      read(streamer, key);
      
      streamer.readArray((char *)&rpos, sizeof(pos_type),1);
      
      if (debug) {
	QDPIO::cout << "Read Key/Position pair. Current position: " << streamer.currentPosition() << endl;
      }
      // Add position to the map
      src_map.insert(std::make_pair(key,rpos));
    }
    QDPUtil::n_uint32_t calc_checksum = streamer.getChecksum();
    QDPUtil::n_uint32_t read_checksum;
    read(streamer, read_checksum);

    if (debug) {
      QDPIO::cout << "Read Map checksum: " << read_checksum << "  Current Position: " << streamer.currentPosition();
    }
    if( read_checksum != calc_checksum ) { 
      QDPIO::cout << "Mismatched Checksums: Expected: " << calc_checksum << " but read " << read_checksum << endl;
      QDP_abort(1);
    }

    if (debug) {
      QDPIO::cout << " Map Checksum OK!" << endl;
    }
  }



  /*!
   * This is a utility function to sync the in memory offset map
   * with the one on the disk, and then close the file for writing.
   * Should not be called by user.
   */
  template<typename K, typename V>
  void
  MapObjectDisk<K,V>::closeWrite(void) 
  {
    switch(state) { 
    case MODIFIED:
    {
      if (debug) {
	QDPIO::cout << "Beginning closeWrite: current position: " << streamer.currentPosition() << endl;
      }

      // Go to end of file
      streamer.seekEnd(0);

      // Take note of current position
      pos_type metadata_start = streamer.currentPosition();
	
      if (debug) {
	QDPIO::cout << "CloseWrite: Metadata starts at position: " << metadata_start << endl;
      }

      // Dump metadata
      writeMapBinary();
	
      // Rewind and Skip header 
      streamer.rewind();
      if (debug) {
	QDPIO::cout << "Rewound file. Current Position: " << streamer.currentPosition() << endl;
      }
      writeSkipHeader();
      if (debug) {
	QDPIO::cout << "Skipped Header. Current Position: " << streamer.currentPosition() << endl;
      }
      // write start position of metadata
      streamer.writeArray((const char *)&metadata_start,sizeof(pos_type),1);
	
      if (debug) {
	QDPIO::cout << "Wrote link to metadata. Current Position: " << streamer.currentPosition() << endl;
      }
	
      // skip to end and close
      streamer.seekEnd(0);
      streamer.flush();
	
      QDPIO::cout << "MapObjectDisk: Closed file " << filename<< " for write access" <<  endl;
    }
    break;
    default:
      errorState("MapObjectDisk: closeWrite() called in an invalid state");
      break;
    }
  }


} // namespace Chroma

#endif
