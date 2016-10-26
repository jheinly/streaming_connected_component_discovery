#pragma once
#ifndef MATH_H
#define MATH_H

namespace core {

namespace math {

template<typename Integer>
Integer divide_and_round_up(
  const Integer value,
  const Integer divisor)
{
  return (value + divisor - 1) / divisor;
}

template<typename Integer>
Integer divide_and_round_down(
  const Integer value,
  const Integer divisor)
{
  return value / divisor;
}

template<typename Integer>
Integer round_up_to_multiple(
  const Integer value,
  const Integer multiple)
{
  return divide_and_round_up(value, multiple) * multiple;
}

template<typename Integer>
Integer round_down_to_multiple(
  const Integer value,
  const Integer multiple)
{
  return divide_and_round_down(value, multiple) * multiple;
}

} // namespace math

} // namespace core

#endif // MATH_H
