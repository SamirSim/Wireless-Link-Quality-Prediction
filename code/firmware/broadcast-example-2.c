/*
 * Copyright (c) 2011, Swedish Institute of Computer Science.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file is part of the Contiki operating system.
 *
 */

#include "contiki.h"
#include "lib/random.h"
#include "sys/ctimer.h"
#include "sys/etimer.h"
#include "net/ip/uip.h"
#include "dev/serial-line.h"
#include "net/ipv6/uip-ds6.h"

#include "simple-udp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define UDP_PORT 1234

#define BEGIN_INTERVAL_SECONDS 10 
#define BEGIN_INTERVAL  (BEGIN_INTERVAL_SECONDS * CLOCK_SECOND)

#define NB_PACKETS 10

#define SEND_BUFFER_SIZE 30
#define SEND_INTERVAL_SECONDS 1
#define SEND_INTERVAL		(SEND_INTERVAL_SECONDS * CLOCK_SECOND)
#define SEND_TIME		(random_rand() % (SEND_INTERVAL))

static struct simple_udp_connection broadcast_connection;

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;
//

/*---------------------------------------------------------------------------*/
PROCESS(broadcast_example_process, "UDP broadcast example process");
AUTOSTART_PROCESSES(&broadcast_example_process);
/*---------------------------------------------------------------------------*/
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

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
static uint32_t 
pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void 
pcg32_srandom_r(pcg32_random_t* rng, 
		uint64_t initstate, 
		uint64_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}
//
/*---------------------------------------------------------------------------*/
PROCESS_THREAD(broadcast_example_process, ev, data)
{
  static unsigned long initstate, initseq;
  static struct etimer periodic_timer;
  char send_buffer[SEND_BUFFER_SIZE];
  static struct etimer begin_timer;
  static struct etimer send_timer;
  static pcg32_random_t rng;
  uip_ipaddr_t addr;
  uint32_t id;
  char *eptr;
  int i;

  PROCESS_BEGIN();

  simple_udp_register(&broadcast_connection, UDP_PORT,
                      NULL, UDP_PORT,
                      receiver);
  clock_init();

  PROCESS_YIELD();
  if (ev == serial_line_event_message) {
    initstate = strtoul((char*)data, &eptr, 10);
  }

  PROCESS_YIELD();
  if (ev == serial_line_event_message) {
    initseq = strtoul((char*)data, &eptr, 10);
  }

  etimer_set(&begin_timer, BEGIN_INTERVAL);
  PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&begin_timer));

  printf("Init state : %lu\n", initstate);
  printf("Init seq : %lu\n", initseq);
  pcg32_srandom_r(&rng, initstate, initseq);

  etimer_set(&periodic_timer, SEND_INTERVAL);
  while(1) {

    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&periodic_timer));
    etimer_reset(&periodic_timer);
    etimer_set(&send_timer, SEND_TIME);

    PROCESS_WAIT_EVENT_UNTIL(etimer_expired(&send_timer));

    //id = nid * clock_seconds();
    id = pcg32_random_r(&rng);
    for (i=0; i<NB_PACKETS; i++) {
    	snprintf(send_buffer, sizeof(uint32_t)*8, "%lx", id+i);
    	printf("SendingBroadcast;%s\n", send_buffer);
    	uip_create_linklocal_allnodes_mcast(&addr);
    	simple_udp_sendto(&broadcast_connection, send_buffer, SEND_BUFFER_SIZE, &addr);
    }

  }

  PROCESS_END();
}
/*---------------------------------------------------------------------------*/
