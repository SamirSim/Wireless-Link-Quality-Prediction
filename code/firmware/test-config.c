#include "contiki.h"
#include "core/net/mac/csma.h"
#include "sys/log.h"
#include "net/ip/uip.h"
#include "net/ipv6/uip-ds6.h"

#include "simple-udp.h"

#define LOG_MODULE "CSMA Test"
#define LOG_LEVEL LOG_LEVEL_INFO

PROCESS(csma_min_be_test_process, "CSMA MIN_BE Test Process");
AUTOSTART_PROCESSES(&csma_min_be_test_process);

static struct simple_udp_connection broadcast_connection;

static void
receiver(struct simple_udp_connection *c,
         const uip_ipaddr_t *sender_addr,
         uint16_t sender_port,
         const uip_ipaddr_t *receiver_addr,
         uint16_t receiver_port,
         const uint8_t *data,
         uint16_t datalen)
{
  printf("Received;%s\n",
         data);
}

PROCESS_THREAD(csma_min_be_test_process, ev, data)
{
  char send_buffer[10];
  static struct etimer timer;
  static uint8_t min_be_value = 3;
  uip_ipaddr_t addr;

  simple_udp_register(&broadcast_connection, 25,
                      NULL, 25,
                      receiver);

  PROCESS_BEGIN();

  // Set a timer to change CSMA_MIN_BE every 10 seconds
  etimer_set(&timer, CLOCK_SECOND * 10);

  while(1) {
    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&timer));

    // Change the CSMA_MIN_BE value
    min_be_value = (min_be_value == 3) ? 5 : 3;
    set_csma_min_be(min_be_value);

    // Log the new CSMA_MIN_BE value
    int new_min_be = get_csma_min_be();
    printf("Changed CSMA_MIN_BE to new value: %u\n", new_min_be);

    snprintf(send_buffer, sizeof(uint32_t)*8, "%lx", 10);
    	printf("SendingBroadcast;%s\n", send_buffer);
    	uip_create_linklocal_allnodes_mcast(&addr);
    	simple_udp_sendto(&broadcast_connection, send_buffer, 10, &addr);

    // Reset the timer
    etimer_reset(&timer);
  }

  PROCESS_END();
}
