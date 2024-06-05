#include <iostream>
#include <chrono>

#include <boost/asio.hpp>
#include <boost/exception/diagnostic_information.hpp> 

#define  VERBOSE
#define BUFFERSIZE 2048

// undef the following for unicast reception
#define MCAST_ADDRESS "239.1.1.22"

using boost::asio::ip::udp;

int main(){
  boost::asio::io_service io_service;
  boost::asio::ip::udp::socket socket(io_service);

  int32_t port = 31200;
  socket.open(boost::asio::ip::udp::v4());
  socket.set_option(udp::socket::reuse_address(true));
  socket.bind(udp::endpoint(boost::asio::ip::address_v4::any(), port));

#ifdef MCAST_ADDRESS
  std::cout << "Joining multicast group " << MCAST_ADDRESS << std::endl;
  socket.set_option(
        boost::asio::ip::multicast::join_group(
          boost::asio::ip::address::from_string(MCAST_ADDRESS)));
#endif

  char buffer[BUFFERSIZE];
  udp::endpoint senderEndpoint;

  unsigned last_sample_offset = 0;
  unsigned last_ms = 0;
  unsigned last_s = 0;
  unsigned expected_increase = 320;
  unsigned nr_packets = 0;
  while (true){
    try{
      auto bytesRead = socket.receive_from(boost::asio::buffer(buffer, BUFFERSIZE), senderEndpoint);
      if (bytesRead > 0) {
        unsigned char* buf = (unsigned char*)buffer;
        unsigned s = {};
        unsigned ms = {};
        unsigned samples = {};

        ms  = (*buf++ & 0x3) << 8;
        ms |= (*buf++);

        samples  = *buf++ << 8;
        samples |= *buf++;

        s  = *buf++ << 24;
        s |= *buf++ << 16;
        s |= *buf++ << 8;
        s |= *buf++;

#ifdef VERBOSE
        if (ms%1000 == 0 && samples == 0) {
         uint64_t now = (duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())).count();
         uint64_t const DIFF_UTC_1970_1980 = 315'964'800U;
         now = (now - (DIFF_UTC_1970_1980*1000)) + 18000;

         int64_t delta = (int64_t)(((int64_t)s*1000L) + ms) - (int64_t)now;

        std::cout << now << ": received " << bytesRead << " bytes for s " << s << ", ms " << ms << 
          ", sample offset " << samples << " from sender " << senderEndpoint.address().to_string() <<
          ", " << delta << " ms delta, packet rate: " << nr_packets << "/ms\n";

        nr_packets = 0;
        }
#endif

        unsigned expected_s = last_s;
        unsigned expected_ms = last_ms;
        unsigned expected_samples = last_sample_offset + expected_increase;

        if (last_sample_offset == 7360) { //15040) {
          expected_samples = 0;
          expected_ms++;
          if (last_ms == 999) {
            expected_ms = 0;
            expected_s++;
          }
        }

        if (ms != expected_ms || s != expected_s || samples != expected_samples) {
          std::cout << "Timestamp discontinuity detected! Expected " 
            << expected_s << "." << std::setfill('0') << std::setw(3) << expected_ms << "." << std::setfill('0') << std::setw(4) << expected_samples <<
            ", got "
            << s << "." << std::setfill('0') << std::setw(3) << ms << "." << std::setfill('0') << std::setw(4) << samples <<
            "\n";
        }

        last_sample_offset = samples;
        last_ms = ms;
        last_s = s;
        nr_packets++;

      }
    }
    catch(const boost::exception &e)
    {
      std::cerr << "Error. Reason: " << boost::diagnostic_information(e) << '\n';
    }
  }
}
